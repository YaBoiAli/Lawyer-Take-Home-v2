document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fuzzySearchForm = document.getElementById('fuzzy-search-form');
    const ragSearchForm = document.getElementById('rag-search-form');
    
    const resultsSection = document.getElementById('results-section');
    const resultsContent = document.getElementById('results-content');
    const spinner = document.getElementById('spinner');

    // --- Event Listeners ---

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);
        const llmProvider = document.getElementById('llm-provider').value;
        formData.append('llm_provider', llmProvider); // Append the selected LLM provider

        await handleRequest('/api/upload-and-extract-data', {
            method: 'POST',
            body: formData
        }, displayUploadResult);
    });

    fuzzySearchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = document.getElementById('fuzzy-query').value;
        await handleRequest('/api/fuzzy-search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        }, displayFuzzyResults);
    });

    ragSearchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = document.getElementById('rag-query').value;
        const fileName = document.getElementById('rag-file-name').value; // Get the optional file name

        const requestBody = { query };
        if (fileName) {
            requestBody.file_name = fileName;
        }

        await handleRequest('/api/rag-search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        }, displayRagResult);
    });

    // --- Core Request Handler ---

    async function handleRequest(url, options, resultHandler) {
        showSpinner();
        resultsContent.innerHTML = ''; // Clear previous results

        try {
            const response = await fetch(url, options);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'An unknown error occurred.');
            }
            
            resultHandler(data);

        } catch (error) {
            displayError(error.message);
        } finally {
            hideSpinner();
        }
    }

    // --- UI Update Functions ---

    function showSpinner() {
        resultsSection.classList.remove('hidden');
        spinner.classList.remove('hidden');
    }

    function hideSpinner() {
        spinner.classList.add('hidden');
    }

    function displayError(message) {
        resultsContent.innerHTML = `<div class="result-item" style="border-color: #dc3545;"><h3>Error</h3><p>${message}</p></div>`;
    }

    function displayUploadResult(data) {
        resultsContent.innerHTML = `<div class="result-item" style="border-color: #28a745;"><h3>Success</h3><p>${data.message}</p></div>`;
    }

    function displayFuzzyResults(data) {
        if (data.length === 0) {
            resultsContent.innerHTML = '<p>No fuzzy search results found.</p>';
            return;
        }

        let html = '<h3>Fuzzy Search Results</h3>';
        data.forEach(item => {
            html += `
                <div class="result-item">
                    <h3>${item.file_name} (Page ${item.page_number})</h3>
                    <p class="result-meta">Similarity Score: ${item.similarity_score}</p>
                    <p>${item.text_preview}</p>
                </div>
            `;
        });
        resultsContent.innerHTML = html;
    }

    function displayRagResult(data) {
        resultsContent.innerHTML = `
            <h3>RAG Answer</h3>
            <div class="rag-answer">${data.answer}</div>
        `;
    }
});
