const sendBtn = document.getElementById('sendBtn');
const userInput = document.getElementById('userInput');
const chatWindow = document.getElementById('chat-window');
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('.section');

function showSection(sectionId) {
    sections.forEach(section => section.classList.remove('active'));
    navLinks.forEach(link => link.classList.remove('active'));
    
    const targetSection = document.getElementById(sectionId);
    const targetLink = document.querySelector(`[data-section="${sectionId}"]`);
    
    if (targetSection) targetSection.classList.add('active');
    if (targetLink) targetLink.classList.add('active');
}

navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const sectionId = link.getAttribute('data-section');
        showSection(sectionId);
        history.pushState(null, '', `#${sectionId}`);
    });
});

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
        const href = anchor.getAttribute('href');
        if (href.startsWith('#') && !anchor.classList.contains('nav-link')) {
            e.preventDefault();
            const sectionId = href.slice(1);
            showSection(sectionId);
            history.pushState(null, '', href);
        }
    });
});

window.addEventListener('hashchange', () => {
    const hash = window.location.hash.slice(1) || 'home';
    showSection(hash);
});

window.addEventListener('load', () => {
    const hash = window.location.hash.slice(1) || 'home';
    showSection(hash);
});

async function handleChat() {
    const prompt = userInput.value.trim();
    if (!prompt) return;

    addMessage(prompt, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';

    const botMsgDiv = addMessage('', 'bot');
    const cursor = document.createElement('span');
    cursor.className = 'typing-cursor';
    botMsgDiv.querySelector('.msg-content').appendChild(cursor);

    sendBtn.disabled = true;

    try {
        const response = await fetch('/streaming', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                prompt: prompt, 
                max_new_tokens: 128 
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        const contentDiv = botMsgDiv.querySelector('.msg-content');

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (let line of lines) {
                if (line.startsWith('data: ')) {
                    const token = line.replace('data: ', '');
                    const cursorEl = contentDiv.querySelector('.typing-cursor');
                    if (cursorEl) {
                        contentDiv.insertBefore(document.createTextNode(token), cursorEl);
                    } else {
                        contentDiv.appendChild(document.createTextNode(token));
                    }
                    chatWindow.scrollTop = chatWindow.scrollHeight;
                }
            }
        }

        const finalCursor = contentDiv.querySelector('.typing-cursor');
        if (finalCursor) finalCursor.remove();

    } catch (err) {
        const contentDiv = botMsgDiv.querySelector('.msg-content');
        contentDiv.textContent = 'Error: Failed to connect to server.';
        console.error(err);
    } finally {
        sendBtn.disabled = false;
    }
}

function addMessage(text, role) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = `msg-avatar ${role}-avatar`;
    
    if (role === 'bot') {
        avatar.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8V4H8"/><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="6" y="14" width="12" height="8" rx="2"/><path d="M12 10v4"/></svg>';
    } else {
        avatar.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
    }
    
    const content = document.createElement('div');
    content.className = 'msg-content';
    content.textContent = text;
    
    div.appendChild(avatar);
    div.appendChild(content);
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    
    return div;
}

sendBtn.addEventListener('click', handleChat);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleChat();
    }
});

userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
});
