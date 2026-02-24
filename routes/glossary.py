"""Tab 6: 専門用語解説"""
from flask import Blueprint, render_template

bp = Blueprint('glossary', __name__, url_prefix='/glossary')


@bp.route('/')
def index():
    return render_template('glossary.html')
