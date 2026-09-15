export function mount(container, host) {
    container.style.marginTop = '40px';
    container.style.padding = '6px 10px';
    container.style.font = '12px monospace';
    container.style.color = '#9ec8e3';
    container.style.display = 'flex';
    container.style.gap = '12px';
    container.style.alignItems = 'center';
    container.style.flexWrap = 'wrap';

    const colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.value = '#000000';
    colorInput.title = 'brush color';
    colorInput.oninput = () =>
        host.eventBus.emit('textureedit:setBrush', { color: hexToRgba(colorInput.value) });

    const radiusInput = document.createElement('input');
    radiusInput.type = 'range';
    radiusInput.min = '2';
    radiusInput.max = '40';
    radiusInput.step = '1';
    radiusInput.value = '14';
    radiusInput.title = 'brush radius';
    radiusInput.oninput = () =>
        host.eventBus.emit('textureedit:setBrush', { radius: parseFloat(radiusInput.value) });

    const opacityInput = document.createElement('input');
    opacityInput.type = 'range';
    opacityInput.min = '0';
    opacityInput.max = '1';
    opacityInput.step = '0.05';
    opacityInput.value = '1';
    opacityInput.title = 'brush opacity';
    opacityInput.oninput = () =>
        host.eventBus.emit('textureedit:setBrush', { opacity: parseFloat(opacityInput.value) });

    const resetBtn = document.createElement('button');
    resetBtn.textContent = 'Reset Canvas';
    resetBtn.style.font = '13px monospace';
    resetBtn.style.padding = '4px 10px';
    resetBtn.style.cursor = 'pointer';
    resetBtn.onclick = () => host.eventBus.emit('textureedit:reset', {});

    const label = (text, el) => {
        const wrap = document.createElement('label');
        wrap.style.display = 'flex';
        wrap.style.alignItems = 'center';
        wrap.style.gap = '4px';
        const span = document.createElement('span');
        span.textContent = text;
        wrap.appendChild(span);
        wrap.appendChild(el);
        return wrap;
    };

    container.appendChild(label('Color', colorInput));
    container.appendChild(label('Radius', radiusInput));
    container.appendChild(label('Opacity', opacityInput));
    container.appendChild(resetBtn);

    return () => { container.innerHTML = ''; };
}

function hexToRgba(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, 1];
}
