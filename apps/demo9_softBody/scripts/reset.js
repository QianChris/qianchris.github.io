// Reset + debug controls (escape-hatch: ScriptComponent scripts run in page
// realm, so window.engine / window.switchApp are reachable globals).
//   R       — lightweight PBD re-seed (destroys GPU buffers; next simulate
//             tick lazily rebuilds the cube from PbdSoftBodyComponent defaults,
//             with a fresh random spawn rotation)
//   Shift+R — full app reload (re-seeds scene + camera + plugins)
//   C       — toggle cluster debug overlay (star lines from COM to members)
//   [ / ]   — cycle selected cluster backward / forward
//
// The keydown handler is tracked on `window` so that scene reloads (via the
// editor's loadJSON, which don't tear down the plugin) remove the previous
// listener before adding a new one — otherwise pressing C would toggle
// debugEnabled twice (old + new listener) and appear to do nothing.

var HANDLER_KEY = '__pbdDemo9Keydown';

export function init() {
    // Remove old listener if a previous init() registered one.
    if (window[HANDLER_KEY]) {
        window.removeEventListener('keydown', window[HANDLER_KEY]);
    }

    var handler = function (e) {
        if (e.key === 'r' || e.key === 'R') {
            if (e.shiftKey) {
                window.switchApp('demo9_softBody');
                return;
            }
            var pbd = window.engine && window.engine.attachments && window.engine.attachments.get('pbd') && window.engine.attachments.get('pbd').obj;
            if (pbd && pbd.clear) {
                pbd.clear();
                console.log('[demo9] PBD re-seeded');
            }
            return;
        }

        if (e.key === 'c' || e.key === 'C') {
            var pbd = window.engine && window.engine.attachments && window.engine.attachments.get('pbd') && window.engine.attachments.get('pbd').obj;
            if (pbd && pbd.setDebugEnabled) {
                var next = !pbd.isDebugEnabled();
                pbd.setDebugEnabled(next);
                var info = pbd.getDebugClusterInfo();
                console.log('[demo9] cluster debug: ' + (next ? 'ON' : 'OFF') +
                    ' — cluster ' + info.cluster + '/' + (info.total - 1) +
                    ' (' + info.memberCount + ' members)');
            }
            return;
        }

        if (e.key === '[' || e.key === ']') {
            var pbd = window.engine && window.engine.attachments && window.engine.attachments.get('pbd') && window.engine.attachments.get('pbd').obj;
            if (pbd && pbd.isDebugEnabled && pbd.isDebugEnabled()) {
                pbd.cycleDebugCluster(e.key === '[' ? -1 : 1);
                var info = pbd.getDebugClusterInfo();
                console.log('[demo9] cluster ' + info.cluster + '/' + (info.total - 1) +
                    ' (' + info.memberCount + ' members)');
            }
            return;
        }
    };

    window[HANDLER_KEY] = handler;
    window.addEventListener('keydown', handler);
}
