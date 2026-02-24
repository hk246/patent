"""Blueprint 登録"""
from flask import Blueprint


def register_blueprints(app):
    from routes.company_patents import bp as company_patents_bp
    from routes.candidate_db import bp as candidate_db_bp
    from routes.clearance import bp as clearance_bp
    from routes.classifier_routes import bp as classifier_bp
    from routes.visualization import bp as visualization_bp
    from routes.glossary import bp as glossary_bp
    from routes.settings import bp as settings_bp
    from routes.report import bp as report_bp

    app.register_blueprint(company_patents_bp)
    app.register_blueprint(candidate_db_bp)
    app.register_blueprint(clearance_bp)
    app.register_blueprint(classifier_bp)
    app.register_blueprint(visualization_bp)
    app.register_blueprint(glossary_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(report_bp)
