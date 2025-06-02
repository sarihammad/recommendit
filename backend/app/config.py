from dotenv import load_dotenv
import os

load_dotenv()

JWT_SECRET = os.environ.get('JWT_SECRET', 'XXX')
JWT_EXPIRATION_MINUTES = os.environ.get('JWT_EXPIRATION_MINUTES', 60)