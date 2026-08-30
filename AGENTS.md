# Project workflow

## Company Profile updates

Whenever a Company Profile is added or updated:

1. Run the relevant focused tests, followed by the full test suite.
2. Start or restart the local Streamlit application with the network access required by its live financial-data providers.
3. Open `http://localhost:8501`, select the affected ticker, and verify that the Company Profile, fundamentals, evidence, and DCF sections render without a data-loading warning.
4. Leave the local application running and give the user the local preview link so they can inspect the update.
5. Keep the updated profile as a research candidate; do not apply it to the Base until the user has reviewed and approved it.
