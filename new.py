import os
import openai
import PyPDF2
import pytesseract
from PIL import Image
import numpy as np
from openai import OpenAI
import logging
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler
from telegram.ext import filters
from sklearn.metrics.pairwise import cosine_similarity
from telegram.ext import CallbackContext, ConversationHandler
import asyncio
import tiktoken
import tempfile
from dotenv import load_dotenv


# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

TOTAL_TOKENS_LIMIT = 15000
total_tokens_spent = 0

#Load environment variables from the .env file
load_dotenv()

telegram_key=os.getenv("TELEGRAM_BOT_TOKEN")

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
     api_key=key,
)


# Tokenizer functions for counting tokens
def count_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Define a function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    logging.info("Extracting text from PDF file")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Define a function to extract text from image files
def extract_text_from_image(image_file):
    logging.info("Extracting text from image file")
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Define a function to create embeddings using OpenAI's model
def create_embeddings(text):
    logging.info("Creating embeddings for text")
    num_tokens = count_tokens(text, "text-embedding-ada-002")
    global total_tokens_spent
    total_tokens_spent += num_tokens
    if total_tokens_spent >= TOTAL_TOKENS_LIMIT:
        logging.error("Token limit reached")
        raise ValueError("Token limit reached. Please enter /start to reset the limit.")
    logging.info(f"Number of tokens used for embeddings (text-embedding-ada-002): {num_tokens}")
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Define a function for finding the best matching context
def find_relevant_context(query_embedding, context_embeddings):
    logging.info("Finding the best matching context")
    similarities = cosine_similarity([query_embedding], context_embeddings)[0]
    best_idx = np.argmax(similarities)
    return best_idx, similarities[best_idx]

# Define a function to handle queries
def answer_question(context, query):
    logging.info("Answering user question")
    query_embedding = create_embeddings(query)
    context_embeddings = [create_embeddings(c) for c in context]
    best_idx, similarity = find_relevant_context(query_embedding, context_embeddings)

    # Добавляем системный промпт
    system_prompt = "Тебя зовут Дарья, ты ИИ ассистент, созданный для помощи по вопросам здорового образа жизни и медицины. Также ты помогаешь пользователям лучше понимать собственные медицинские анализы и исследования. Отвечай только на вопросы, связанные с медицинской тематикой. Если вопрос не связан с медициной, вежливо откажись отвечать"

    if similarity > 0.6:  # Adjust the threshold as needed
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context: {context[best_idx]}"},
            {"role": "user", "content": query}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

    # Count tokens for messages
    messages_text = " ".join([msg["content"] for msg in messages])
    num_tokens = count_tokens(messages_text, "gpt-4o")
    global total_tokens_spent
    total_tokens_spent += num_tokens
    if total_tokens_spent >= TOTAL_TOKENS_LIMIT:
        logging.error("Token limit reached")
        raise ValueError("Token limit reached. Please enter /start to reset the limit.")
    logging.info(f"Number of tokens used for GPT-4o completion: {num_tokens}")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Define the Telegram bot functions
context_storage = []  # A list to store extracted contexts from the uploaded files
context_embeddings_storage = []  # A list to store embeddings of the contexts

async def start(update, context: CallbackContext):
    logging.info("Handling /start command")
    global total_tokens_spent
    total_tokens_spent = 0  # Reset the token count
    keyboard = [
        [InlineKeyboardButton("Сделать резюме", callback_data='summarize')],
        [KeyboardButton("Мои токены")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text('Привет! Меня зовут Дарья, я ИИ ассистент, созданный для помощи по вопросам здорового образа жизни и медицины! Ты можешь задать интересующий тебя вопрос по медицине или по своим анализам', reply_markup=reply_markup)

# Function to handle file uploads (PDF, JPEG, PNG, JPG)
async def handle_file(update, context: CallbackContext):
    logging.info(f"Received a file: {update.message.document.file_name if update.message.document else 'photo'}")
    logging.info("Handling file upload")
    file = await (update.message.document.get_file() if update.message.document else update.message.photo[-1].get_file())
    if file.file_size > 3 * 1024 * 1024:  # Limit file size to 3 MB
        logging.warning("File size exceeds limit")
        await update.message.reply_text('Файл слишком большой. Максимальный размер файла - 3 МБ.')
        return

    file_extension = 'jpg' if update.message.photo else file.file_path.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
        await file.download_to_drive(temp_file.name)
        logging.info(f"File downloaded to {temp_file.name}")

        if file_extension == 'pdf':
            text = extract_text_from_pdf(temp_file.name)
        elif file_extension in ['jpeg', 'png', 'jpg']:
            text = extract_text_from_image(temp_file.name)
        else:
            logging.warning("Unsupported file format")
            await update.message.reply_text('Неподдерживаемый формат файла.')
            return

    try:
        context_storage.append(text)
        embedding = create_embeddings(text)
        context_embeddings_storage.append(embedding)
        await update.message.reply_text('Файл загружен успешно! Теперь вы можете задавать вопросы по его содержимому.')
    except ValueError as e:
        logging.error(f"Error while processing file: {e}")
        await update.message.reply_text(str(e))

# Function to handle user text queries
async def handle_query(update, context: CallbackContext):
    logging.info("Handling user text query")
    query = update.message.text
    global total_tokens_spent
    if query == "Мои токены":
        tokens_left = max(0, TOTAL_TOKENS_LIMIT - total_tokens_spent)
        await update.message.reply_text(f'Суммарный объем потраченных токенов: {total_tokens_spent}\nОсталось токенов до лимита: {tokens_left}')
        return
    
    try:
        # Count tokens for the user's query
        num_tokens = count_tokens(query, "gpt-4o")
        total_tokens_spent += num_tokens
        if total_tokens_spent >= TOTAL_TOKENS_LIMIT:
            logging.error("Token limit reached")
            raise ValueError("Token limit reached. Please enter /start to reset the limit.")
        
        if context_storage:
            query_embedding = create_embeddings(query)
            best_idx, similarity = find_relevant_context(query_embedding, context_embeddings_storage)
            if similarity > 0.6:
                answer = answer_question([context_storage[best_idx]], query)
            else:
                answer = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": query}],
                    max_tokens=700
                ).choices[0].message.content.strip()
        else:
            # No files available, just use the model to answer the question
            answer = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": query}],
                max_tokens=700
            ).choices[0].message.content.strip()
        await update.message.reply_text(answer)
    except ValueError as e:
        logging.error(f"Error while handling query: {e}")
        await update.message.reply_text(str(e))

# Function to summarize all uploaded files
async def summarize(update, context: CallbackContext):
    logging.info("Handling summarize request")
    if context_storage:
        combined_text = "\n".join(context_storage)
        try:
            summary = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Пожалуйста, сделайте краткое резюме следующего текста: " + combined_text}],
                max_tokens=700
            ).choices[0].message.content.strip()
            await update.callback_query.message.reply_text(summary)
        except ValueError as e:
            logging.error(f"Error while summarizing: {e}")
            await update.callback_query.message.reply_text(str(e))
    else:
        logging.warning("No documents available for summarization")
        await update.callback_query.message.reply_text('Нет загруженных документов для создания резюме.')

# Function to handle button clicks
async def button(update, context: CallbackContext):
    logging.info("Handling button click")
    query = update.callback_query
    await query.answer()
    if query.data == 'summarize':
        await summarize(update, context)

# Main function to set up the bot
def main():
    logging.info("Starting the bot")
    application = ApplicationBuilder().token(telegram_key).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    application.add_handler(MessageHandler(filters.PHOTO, handle_file))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    application.run_polling()

if __name__ == '__main__':
    logging.info("Bot is running")
    main()
