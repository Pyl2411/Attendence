import os


def _is_streamlit_runtime():
    return bool(
        os.environ.get("STREAMLIT_SERVER_PORT")
        or os.environ.get("STREAMLIT_RUNTIME")
        or os.environ.get("STREAMLIT_SERVER_HEADLESS")
    )


if _is_streamlit_runtime():
    from app import main as run_app
else:
    from src.gui_app import main as run_app


if __name__ == "__main__":
    run_app()
