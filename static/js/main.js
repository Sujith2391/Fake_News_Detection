
// Theme Logic
const themeBtn = document.querySelector('.theme-toggle');
const html = document.documentElement;
const icon = themeBtn.querySelector('i');

// Check saved preference
const savedTheme = localStorage.getItem('theme') || 'dark';
html.setAttribute('data-theme', savedTheme);
updateIcon(savedTheme);

themeBtn.addEventListener('click', () => {
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateIcon(newTheme);
});

function updateIcon(theme) {
    if (theme === 'light') {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    } else {
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    }
}

// Analysis Logic (Only if on analyzer page)
if (document.getElementById('newsInput')) {
    document.getElementById('newsInput').addEventListener('input', function (e) {
        document.getElementById('charCount').textContent = e.target.value.length + " characters";
    });

    // Make functions globally available for onclick events
    window.analyzeNews = async function () {
        const text = document.getElementById('newsInput').value;
        const btn = document.getElementById('analyzeBtn');

        if (!text.trim()) {
            alert("Please enter some text to analyze.");
            return;
        }

        // Set loading state
        btn.disabled = true;
        btn.innerHTML = '<span>Analyzing...</span><i class="fa-solid fa-spinner fa-spin"></i>';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            const data = await response.json();

            if (response.ok) {
                showResult(data);
            } else {
                alert("Error: " + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert("An error occurred while communicating with the server.");
        } finally {
            // Reset button
            btn.disabled = false;
            btn.innerHTML = '<span>Analyze content</span><i class="fa-solid fa-wand-magic-sparkles"></i>';
        }
    }

    function showResult(data) {
        const container = document.getElementById('resultContainer');
        const title = document.getElementById('verdictTitle');
        const desc = document.getElementById('verdictDesc');
        const score = document.getElementById('confidenceScore');
        const icon = document.getElementById('verdictIcon');

        container.classList.remove('hidden');
        container.classList.remove('result-fake', 'result-real');

        if (data.is_fake) {
            container.classList.add('result-fake');
            title.textContent = "Potential Fake News";
            desc.textContent = "Our model has flagged this content as potentially misleading or false.";
            icon.innerHTML = '<i class="fa-solid fa-circle-exclamation"></i>';
        } else {
            container.classList.add('result-real');
            title.textContent = "Likely Authentic";
            desc.textContent = "This content appears to be consistent with reliable news patterns.";
            icon.innerHTML = '<i class="fa-solid fa-shield-check"></i>';
        }

        score.textContent = data.confidence;
    }
}
