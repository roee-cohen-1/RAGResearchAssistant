import os
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner

from rag import assistant, generate_instructions, augmentation


class Message:

    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.feedback = None
        self.context_renderable = None


console = Console()
instructions = None
chat = []
context = []
subject = None


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def handle_user_feedback(feedback):
    feedback = feedback.replace('/feedback ', '')
    for i in reversed(range(len(chat))):
        if chat[i].role == 'assistant':
            chat[i].feedback = feedback
            break
    with Live(Spinner('dots'), console=console, refresh_per_second=8) as live:
        new_ins = generate_instructions(chat, previous_instructions=instructions)
        live.update('')
    return new_ins


def handle_user_message(message):
    """
    Renders the augmentation and generation of the assistant's response
    """
    global context, subject
    console.print('\n')
    chat.append(Message('you', message))
    with Live(
            Spinner('dots', text=''),
            console=console,
            refresh_per_second=10
    ) as live:
        for result in augmentation(subject, chat, context):
            # end of augmentation
            if type(result) == tuple:
                subject, context, chunks = result
                # save table to be displayed later
                context_ren = live.get_renderable()
                break
            else:
                live.update(result)
    text = '\n'
    for chunk in assistant(chat, chunks, instructions):
        if chunk:
            if chunk.type == "content-delta":
                text += chunk.delta.message.content.text
                console.print(chunk.delta.message.content.text, end='')
    chat.append(Message('assistant', text))
    chat[-1].context_renderable = context_ren


def chat_page():
    """
    Renders the chat
    """
    clear_screen()
    console.print('Welcome to your research assistant for the fields of AI!\n', style='cyan bold')
    console.print('[purple]assistant>[/purple] What would you like to research today?\n')
    for message in chat:
        if message.role == 'you':
            console.print(f'[purple]{message.role}>[/purple]', end=' ')
            console.print(f'{message.content}', style='cyan')
        elif message.role == 'assistant':
            console.print(f'[purple]context>[/purple]\n')
            console.print(message.context_renderable)
            console.print('\n')
            console.print(f'[purple]{message.role}>[/purple]\n')
            md = Markdown(message.content)
            console.print(md, style='white')
        else:
            md = Markdown(message.content)
            console.print(md, style='bold red')
        console.print('\n')
    return console.input(f'[purple]you>[/purple] ')


def main():
    """
    The main loop of the application
    """
    global instructions
    while True:
        command = chat_page()
        if command.strip() == '':
            pass
        if command.startswith('/'):
            if command == '/ins' and instructions is not None:
                chat.append(Message('system', instructions))
            if command.startswith('/feedback'):
                instructions = handle_user_feedback(command)
            if command == '/exit':
                return
        else:
            handle_user_message(command)


if __name__ == '__main__':
    main()
