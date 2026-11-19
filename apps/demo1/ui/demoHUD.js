export function mount(container, host) {
    container.style.marginTop = '40px';
    container.style.padding = '6px 10px';

    const btn = document.createElement('button');
    btn.textContent = 'Click Me';
    const msg = document.createElement('div');
    msg.style.color = '#7ec8e3';
    msg.style.font = '12px monospace';
    msg.style.marginTop = '6px';

    container.appendChild(btn);
    container.appendChild(msg);

    const off = host.eventBus.on('toast:show', (text) => {
        msg.textContent = String(text);
    });
    btn.onclick = () => host.eventBus.emit('toast:show', 'Hello from App UI!');

    return () => { off(); btn.remove(); msg.remove(); };
}
