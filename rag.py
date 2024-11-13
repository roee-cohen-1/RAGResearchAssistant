from dotenv import load_dotenv
import numpy as np
import cohere
from rich.spinner import Spinner
from rich.table import Table
import io
import os
import arxiv2text
import arxiv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google.cloud import storage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage

load_dotenv()

bucket = storage.Client().bucket('arxiv-dataset')

_pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
_model = SentenceTransformer('all-MiniLM-L6-v2')
_client = arxiv.Client()

co = cohere.ClientV2(api_key=os.environ['COHERE_API_KEY'])


def _get_threshold(arr):
    """
    Finds a threshold for an array of relevance scores.
    Normally it is the .9th quantile but if the part above it
    is larger than the part below it, .5 is returned.
    """
    upper = np.quantile(arr, q=.9)
    if 1 - upper > upper:
        return .5
    return upper


def _chat_to_messages(chat):
    """
    Transforms the chat to Cohere readable data
    """
    return [
        {
            'role': 'user' if msg.role == 'you' else 'assistant',
            'content': msg.content
        }
        for msg in chat
    ]


def _user_messages_to_text(chat):
    text = '\n'.join(
        f'Message: {msg.content}'
        for msg in chat if msg.role == 'you'
    )
    return text


def _chat_to_text(chat, feedback=True):
    text = '\n'.join(
        f'''
        Role: {'user' if msg.role == 'you' else 'assistant'}
        Content: {msg.content}
        Feedback: {msg.feedback}
        '''
        if feedback and msg.feedback is not None else
        f'''
        Role: {'user' if msg.role == 'you' else 'assistant'}
        Content: {msg.content}
        '''
        for msg in chat if msg.role in ('you', 'assistant')
    )
    return text


def _context_to_text(subject, context):
    """
    The documents are chunked and the chunks are re-ranked
    """
    chunks = []
    for document in context:
        if document.selected and document.full is not None:
            parts = document.full.split('\n\n')
            for part in parts:
                chunks.append({
                    'id': document.get_id(),
                    'title': document.title,
                    'snippet': part
                })
    result = co.rerank(
        model='rerank-english-v3.0',
        query=subject,
        documents=[item['snippet'] for item in chunks]
    )
    threshold = _get_threshold([index.relevance_score for index in result.results])
    count = min(10, sum(1 if index.relevance_score >= threshold else 0 for index in result.results))
    text = ''
    for index in result.results[:count]:
        text += f'Title: {chunks[index.index]["title"]}\n'
        text += f'Snippet: {chunks[index.index]["snippet"]}\n'
    return text, [chunks[index.index]['id'] for index in result.results[:count]]


def generate_instructions(chat, previous_instructions):
    ins = """
    You are given a conversation between a chatbot assistant and 
    a user along with user feedbacks on the chatbot's responses.
    You are also given the current instructions for the assistant.
    Analyse the chat and the user feedback and write style instructions
    for the chatbot to better suit the user.
    Completely ignore the subject of the conversation! Only style instructions!
    """
    chat = _chat_to_text(chat)
    chat = f'\nPrevious Instructions: {previous_instructions}\n' + chat
    res = co.chat(
        model="command-r-plus-08-2024",
        messages=[
            {'role': 'system', 'content': ins},
            {'role': 'user', 'content': chat},
        ],
    )
    return res.message.content[0].text


def assistant(chat, chunks, instruction):
    messages = _chat_to_messages(chat)
    instruction = f'''
        {instruction if instruction is not None else ''}

        Instructions:
        You are a research assistants for the field of AI.
        You are given some context from external sources.
        Use the context if you think that it is relevant.

        Context:
        {chunks}

    '''
    messages = [{'role': 'system', 'content': instruction}] + messages
    return co.chat_stream(
        model="command-r-plus-08-2024",
        messages=messages
    )


def find_ai_subject(subject, chat):
    if subject is None:
        instructions = """
        Analyse the user message sent to his research assistant for the field if AI.
        The messages are written from oldest to newest.
        Find the current subject in the field of AI that the user wants to research.
        Write it in a few keywords that can later be used to search relevant articles.
        If there is not a clear subject, write 'no AI in chat' and nothing else.
        If you found keywords, write only them and nothing else.
        """
    else:
        instructions = f"""
        Analyse the user message sent to his research assistant for the field if AI.
        The messages are written from oldest to newest.
        The subject of research found before the last user message was: "{subject}".
        Refine the subject to the current subject of the chat. 
        If you think the subject remained the same, write 'no change' and nothing else.
        If you think the subject can be refined, write it in a few keywords that can later be used to search relevant articles.
        If you found keywords, write only them and nothing else.
        """
    res = co.chat(
        model="command-r-08-2024",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": _user_messages_to_text(chat)},
        ]
    )
    response = res.message.content[0].text
    if response.lower() == 'no ai in chat':
        return None
    elif response.lower() == 'no change':
        return subject
    return response


def find_context(subject):
    context = fetch_articles(co, subject)
    scores = [article.relevance for article in context]
    threshold = _get_threshold(scores)
    count = min(5, sum(1 if score >= threshold else 0 for score in scores))
    for i in range(count):
        context[i].selected = True
    return context, threshold


def download_selected_articles(context):
    for i in range(len(context)):
        if context[i].selected and context[i].full is None:
            context[i].full = _fetch_full_text(co, context[i])
            yield context[i].title


def augmentation(subject, chat, context):
    yield Spinner('dots')
    new_subject = find_ai_subject(subject, chat)
    if new_subject == subject:
        documents, chunk_counts = _context_to_text(subject, context)
        yield subject, context, documents
    else:
        subject = new_subject
        yield Spinner('dots', text='searching articles on arXiv')
        context, threshold = find_context(subject)

        def render(counts=None):
            table = Table(
                title=f'arXiv Papers for The Subject: {subject}',
                collapse_padding=True,
                padding=2
            )
            table.add_column('', justify='center')
            table.add_column('ID', justify='center')
            table.add_column('Rel', justify='center')
            table.add_column('Chunks', justify='center')
            table.add_column('Title', justify='center')
            for doc in context:
                if not doc.selected:
                    continue
                if doc.full is None:
                    table.add_row(
                        Spinner('dots'),
                        doc.get_id(),
                        f'{round(doc.relevance * 100):.0f}%',
                        '',
                        doc.title
                    )
                    continue
                if counts is not None:
                    count = sum(1 if id_ == doc.get_id() else 0 for id_ in counts)
                    table.add_row(
                        '✓',
                        doc.get_id(),
                        f'{round(doc.relevance * 100):.0f}%',
                        f'{count} / {len(counts)}',
                        doc.title
                    )
                else:
                    table.add_row(
                        '✓',
                        doc.get_id(),
                        f'{round(doc.relevance * 100):.0f}%',
                        '',
                        doc.title
                    )
            return table

        for _ in download_selected_articles(context):
            yield render()

        documents, chunk_counts = _context_to_text(subject, context)

        yield render(counts=chunk_counts)

        yield subject, context, documents


class Document:

    def __init__(self, id_, title, abstract):
        self.id = id_
        self.title = title
        self.abstract = abstract
        self.selected = False
        self.full = None
        self.explanation = None
        self.relevance = None

    def get_id(self):
        id_ = self.id.split('/')[-1]
        if 'v' in id_:
            id_ = id_.split('v')[0]
        return id_


def fetch_articles(co, query, top_k=50):
    index = _pc.Index('arxiv-abs')
    emb = _model.encode(query).tolist()
    results = index.query(top_k=top_k, vector=emb, include_metadata=True)
    results = {m['id']: m for m in results['matches']}
    ids = list(results.keys())
    # fetch abstract text
    articles = _client.results(search=arxiv.Search(id_list=ids))
    articles = [
        Document(article.entry_id, article.title, article.summary)
        for article in articles
    ]
    # re-rank the abstracts
    result = co.rerank(
        model='rerank-english-v3.0',
        query=query,
        documents=[doc.abstract for doc in articles]
    )
    # save the relevance score
    for index in result.results:
        articles[index.index].relevance = index.relevance_score
    return [articles[r.index] for r in result.results]


def _pdf_to_text(pdf_bytes) -> str:
    """
    code found online for parsing pdf to text using pdfminer
    """
    pdf_file = io.BytesIO(pdf_bytes)

    resource_manager = PDFResourceManager()
    text_stream = io.StringIO()
    laparams = LAParams()

    device = TextConverter(resource_manager, text_stream, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, device)
    for page in PDFPage.get_pages(pdf_file):
        interpreter.process_page(page)

    extracted_text = text_stream.getvalue()
    text_stream.close()

    return extracted_text


def _parse(co, text):
    """
    paper parsing by Cohere
    """
    instructions = """
    You are given the text of an article from arXiv.
    Please clean it and keep only the relevant information.
    Split the text into paragraphs with an empty line separating the paragraphs.
    Remove any non relevant information such as bibliography.
    """
    try:
        res = co.chat(
            model="command-r-08-2024",
            messages=[
                {'role': 'system', 'content': instructions},
                {'role': 'user', 'content': text}
            ]
        )
        return res.message.content[0].text
    except:
        return 'Empty Document'


def _check_cache(id_):
    """
    Caching mechanism
    """
    path = os.path.join('cache', f'{id_}.txt')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    return None


def _cache(id_, text):
    """
    Caching mechanism
    """
    path = os.path.join('cache', f'{id_}.txt')
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)


def _fetch_full_text(co, article):
    id_ = article.id.split('/')[-1]
    # check cache first
    text = _check_cache(id_)
    if text is not None:
        return text
    # check GCP cloud
    blob_name = f"arxiv/arxiv/pdf/{id_[:4]}/{id_}.pdf"
    blob = bucket.blob(blob_name)
    if blob.exists():
        bytes = blob.download_as_bytes()
        with open(f'cache/{id_}.pdf', 'wb') as file:
            file.write(bytes)
        text = _pdf_to_text(bytes)
    else:
        # fetch directly from arXiv
        try:
            text = arxiv2text.arxiv_to_text(article.id.replace('abs', 'pdf'))
        except:
            text = 'Empty Document'
    # preprocessing step: remove unnecessary line drops
    text = text.replace('\n\n', '!@#$%^&*').replace('\n', ' ').replace('!@#$%^&*', '\n')
    text = _parse(co, text)
    _cache(id_, text)
    return text
