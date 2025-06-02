from flask import Blueprint, request, jsonify
from app.services.auth_service import signup_user, login_user, get_user_info
from app.utils.auth_utils import token_required

auth_routes = Blueprint('auth_routes', __name__, url_prefix='/auth')

@auth_routes.route('/signup', methods=['POST'])
def signup():
    data = request.json
    response, status = signup_user(data.get('email'), data.get('password'))
    return jsonify(response), status

@auth_routes.route('/login', methods=['POST'])
def login():
    data = request.json
    response, status = login_user(data.get('email'), data.get('password'))
    return jsonify(response), status

@auth_routes.route('/me', methods=['GET'])
@token_required
def get_me():
    user_id = request.user_id
    response, status = get_user_info(user_id)
    return jsonify(response), status