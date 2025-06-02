from werkzeug.security import generate_password_hash, check_password_hash
from app.db import get_db
from app.utils.auth_utils import generate_token

def signup_user(email, password):
    if not email or not password:
        return {'error': 'Email and password are required'}, 400

    db = get_db()
    try:
        db.execute(
            'INSERT INTO user (email, password) VALUES (?, ?)',
            (email, generate_password_hash(password))
        )
        db.commit()
    except db.IntegrityError:
        return {'error': 'User already exists'}, 409

    return {'message': 'User created successfully'}, 201

def login_user(email, password):
    if not email or not password:
        return {'error': 'Email and password are required'}, 400

    db = get_db()
    user = db.execute(
        'SELECT * FROM user WHERE email = ?', (email,)
    ).fetchone()

    if user is None or not check_password_hash(user['password'], password):
        return {'error': 'Invalid credentials'}, 401

    token = generate_token(user['id'])
    return {'token': token, 'user_id': user['id']}, 200

def get_user_info(user_id):
    db = get_db()
    user = db.execute(
        "SELECT id, email FROM user WHERE id = ?", (user_id,)
    ).fetchone()
    if user:
        return dict(user), 200
    return {'error': 'User not found'}, 404