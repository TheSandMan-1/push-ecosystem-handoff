import os

from rolescout.config import _load_dotenv, get_settings


def test_dotenv_strips_trailing_comments_but_keeps_quoted_hashes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "ROLESCOUT_LOCAL_LLM_NUM_CTX=16384    # context window\n"
        "ROLESCOUT_VERIFY_SAMPLE_RATE=0.1 # QA sample\n"
        'SMTP_PASSWORD="pa ss#word"\n'
        "ROLESCOUT_LOCAL_LLM_BASE_URL=http://localhost:1234/v1\n"
        "# a full-line comment\n"
    )
    for key in ("ROLESCOUT_LOCAL_LLM_NUM_CTX", "ROLESCOUT_VERIFY_SAMPLE_RATE", "SMTP_PASSWORD", "ROLESCOUT_LOCAL_LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    _load_dotenv(env)
    assert os.environ["ROLESCOUT_LOCAL_LLM_NUM_CTX"] == "16384"
    assert os.environ["SMTP_PASSWORD"] == "pa ss#word"
    get_settings.cache_clear()
    s = get_settings()
    assert s.local_llm_num_ctx == 16384 and s.verify_sample_rate == 0.1
    assert s.local_llm_base_url == "http://localhost:1234/v1"
    for key in ("ROLESCOUT_LOCAL_LLM_NUM_CTX", "ROLESCOUT_VERIFY_SAMPLE_RATE", "SMTP_PASSWORD", "ROLESCOUT_LOCAL_LLM_BASE_URL"):
        os.environ.pop(key, None)
