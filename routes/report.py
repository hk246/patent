"""使い方ガイド"""
from flask import Blueprint, render_template

bp = Blueprint('report', __name__)


@bp.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')
