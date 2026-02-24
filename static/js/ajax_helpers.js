/**
 * AJAX helpers with loading spinner management
 */

function showLoading(message) {
    const overlay = document.getElementById('loading-overlay');
    const msg = document.getElementById('loading-message');
    if (msg) msg.textContent = message || '処理中...';
    if (overlay) overlay.classList.remove('hidden');
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.classList.add('hidden');
}

function showFlash(message, category) {
    category = category || 'info';
    const colors = {
        success: 'bg-emerald-900/30 border-emerald-700/50 text-emerald-300',
        error: 'bg-red-900/30 border-red-700/50 text-red-300',
        warning: 'bg-amber-900/30 border-amber-700/50 text-amber-300',
        info: 'bg-sky-900/30 border-sky-700/50 text-sky-300',
    };
    const div = document.createElement('div');
    div.className = `flash-msg px-4 py-3 rounded-lg border flex items-center justify-between text-sm ${colors[category] || colors.info}`;
    div.innerHTML = `<span>${message}</span><button onclick="this.parentElement.remove()" class="ml-4 text-gray-500 hover:text-gray-300">&times;</button>`;

    let container = document.querySelector('main .space-y-2');
    if (!container) {
        container = document.createElement('div');
        container.className = 'space-y-2 mb-4';
        const main = document.querySelector('main');
        main.insertBefore(container, main.firstChild);
    }
    container.appendChild(div);

    setTimeout(() => { if (div.parentElement) div.remove(); }, 8000);
}

/**
 * Generic AJAX POST. Returns parsed JSON.
 * @param {string} url
 * @param {FormData|Object} data - FormData or plain object (sent as JSON)
 * @param {Object} options - { loadingMessage: string }
 */
async function ajaxPost(url, data, options) {
    options = options || {};
    showLoading(options.loadingMessage);
    try {
        let fetchOptions = { method: 'POST' };
        if (data instanceof FormData) {
            fetchOptions.body = data;
        } else {
            fetchOptions.body = JSON.stringify(data);
            fetchOptions.headers = { 'Content-Type': 'application/json' };
        }
        const resp = await fetch(url, fetchOptions);
        const result = await resp.json();
        hideLoading();
        if (result.error) {
            showFlash(result.error, 'error');
        }
        return result;
    } catch (err) {
        hideLoading();
        showFlash('通信エラー: ' + err.message, 'error');
        throw err;
    }
}

/**
 * Insert HTML that may contain <script> tags and execute them.
 * Normal innerHTML assignment does not execute scripts.
 */
function insertHTMLWithScripts(container, html) {
    container.innerHTML = html;
    // Find all script tags and re-create them so the browser executes them
    container.querySelectorAll('script').forEach(function(oldScript) {
        var newScript = document.createElement('script');
        if (oldScript.src) {
            newScript.src = oldScript.src;
        } else {
            newScript.textContent = oldScript.textContent;
        }
        oldScript.parentNode.replaceChild(newScript, oldScript);
    });
}

/**
 * Collect settings form data and save via AJAX
 */
function getSettingsFormData() {
    const form = document.getElementById('settings-form');
    if (!form) return {};
    const data = {};
    const formData = new FormData(form);
    for (const [key, val] of formData.entries()) {
        data[key] = val;
    }
    // Handle unchecked checkboxes
    ['use_lsa', 'use_stop_words', 'use_claim_weight'].forEach(key => {
        if (!(key in data)) data[key] = 'false';
    });
    return data;
}

async function saveSettings() {
    const data = getSettingsFormData();
    try {
        await fetch('/settings/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
    } catch (e) {
        // Silent fail for auto-save
    }
}
