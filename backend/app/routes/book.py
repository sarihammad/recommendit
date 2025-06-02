from app.services.db_service import get_liked_books, store_feedback
from flask import Blueprint, request, jsonify
from app.services.book_service import recommend_from_likes, recommend_hybrid, top_books, recommend_collaboratively
from app.utils.auth_utils import token_required

book_routes = Blueprint('book_routes', __name__, url_prefix='/books')

@book_routes.route('/feedback', methods=['POST'])
@token_required
def store_feedback_route():
    data = request.json
    user_id = request.user_id
    book_id = data.get('book_id')
    liked = data.get('liked')

    if not user_id or not book_id or liked is None:
        return jsonify({'error': 'Missing fields'}), 400

    store_feedback(user_id, book_id, liked)
    return jsonify({'message': 'Feedback recorded'}), 200

@book_routes.route('/recommendations', methods=['GET'])
@token_required
def recommend_for_user():
    user_id = request.user_id
    liked_book_ids = get_liked_books(user_id)

    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    offset = (page - 1) * limit

    all_recommendations = recommend_hybrid(user_id, liked_book_ids, top_n=100)
    paginated = all_recommendations.iloc[offset:offset + limit]
    return jsonify(paginated.to_dict(orient='records'))


@book_routes.route('/popular', methods=['GET'])
def top_books_route():
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    offset = (page - 1) * limit

    books_df = top_books(n=100)
    paginated = books_df.iloc[offset:offset + limit]
    return jsonify(paginated.to_dict(orient='records'))