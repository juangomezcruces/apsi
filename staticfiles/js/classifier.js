// Example text filling function
function fillExample(text) {
    const textInput = document.getElementById('id_text');
    if (textInput) {
        textInput.value = text.trim();
        autoResizeTextarea(textInput);
        updateCharacterCount();
        textInput.focus();
        textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Form validation
function validateForm() {
    const textInput = document.getElementById('id_text');
    const text = textInput.value.trim();
    
    if (!text) {
        alert('Please enter some text to analyze.');
        textInput.focus();
        return false;
    }
    
    if (text.length > 2000) {
        alert('Text is too long. Please limit to 2000 characters.');
        textInput.focus();
        return false;
    }
    
    return true;
}

// Character counter
function updateCharacterCount() {
    const textInput = document.getElementById('id_text');
    const counter = document.getElementById('char-counter');
    
    if (textInput && counter) {
        const remaining = 2000 - textInput.value.length;
        counter.textContent = `${remaining} characters remaining`;
        
        if (remaining < 100) {
            counter.className = 'text-warning';
        } else if (remaining < 0) {
            counter.className = 'text-danger';
        } else {
            counter.className = 'text-muted';
        }
    }
}

// Auto-resize textarea
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Add form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            if (!validateForm()) {
                e.preventDefault();
            }
        });
    }
    
    // Add character counter
    const textInput = document.getElementById('id_text');
    if (textInput) {
        // Create counter element
        const counter = document.createElement('div');
        counter.id = 'char-counter';
        counter.className = 'text-muted small mt-1';
        textInput.parentElement.appendChild(counter);
        
        // Update counter on input
        textInput.addEventListener('input', updateCharacterCount);
        textInput.addEventListener('input', function() {
            autoResizeTextarea(this);
        });
        
        // Initial update
        updateCharacterCount();
    }
});

// API helper function
async function classifyTextAPI(text) {
    try {
        const response = await fetch('/api/classify/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Classification failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}