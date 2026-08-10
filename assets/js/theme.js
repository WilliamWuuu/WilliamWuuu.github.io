(function () {
    'use strict';

    var STORAGE_KEY = 'theme';
    var MODES = ['light', 'dark', 'system'];
    var LIGHT_HIGHLIGHT_THEME = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-light.min.css';
    var DARK_HIGHLIGHT_THEME = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css';
    var systemPreference = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
    var currentMode = readStoredMode();
    var switcher = null;
    var thumb = null;
    var options = [];
    var dragState = null;
    var suppressClick = false;

    function readStoredMode() {
        try {
            var storedMode = window.localStorage.getItem(STORAGE_KEY);
            return MODES.indexOf(storedMode) !== -1 ? storedMode : 'light';
        } catch (error) {
            return 'light';
        }
    }

    function storeMode(mode) {
        try {
            window.localStorage.setItem(STORAGE_KEY, mode);
        } catch (error) {
            // The theme still works for this page if storage is unavailable.
        }
    }

    function resolveTheme(mode) {
        if (mode === 'system') {
            return systemPreference && systemPreference.matches ? 'dark' : 'light';
        }
        return mode;
    }

    function updateHighlightTheme(theme) {
        var highlightTheme = document.getElementById('hljs-theme');
        if (highlightTheme) {
            highlightTheme.href = theme === 'dark' ? DARK_HIGHLIGHT_THEME : LIGHT_HIGHLIGHT_THEME;
        }
    }

    function renderSwitcher() {
        if (!switcher) return;

        switcher.setAttribute('data-mode', currentMode);
        options.forEach(function (option) {
            var isActive = option.getAttribute('data-theme-mode') === currentMode;
            option.setAttribute('aria-checked', isActive ? 'true' : 'false');
            option.setAttribute('tabindex', isActive ? '0' : '-1');
            option.classList.toggle('is-active', isActive);
        });
    }

    function applyMode(mode, persist) {
        if (MODES.indexOf(mode) === -1) return;

        currentMode = mode;
        var resolvedTheme = resolveTheme(mode);
        document.documentElement.setAttribute('data-theme-mode', mode);
        document.documentElement.setAttribute('data-theme', resolvedTheme);
        updateHighlightTheme(resolvedTheme);
        renderSwitcher();

        if (persist) {
            storeMode(mode);
        }
    }

    function modeIndex(mode) {
        return Math.max(0, MODES.indexOf(mode));
    }

    function getGeometry() {
        var switcherRect = switcher.getBoundingClientRect();
        var firstOptionRect = options[0].getBoundingClientRect();
        var segmentWidth = firstOptionRect.width;
        return {
            trackLeft: firstOptionRect.left - switcherRect.left,
            segmentWidth: segmentWidth,
            maxOffset: segmentWidth * (MODES.length - 1)
        };
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function finishDrag(targetIndex, animate) {
        if (!dragState) return;

        var pointerId = dragState.pointerId;
        dragState = null;
        applyMode(MODES[targetIndex], true);

        if (switcher.hasPointerCapture && switcher.hasPointerCapture(pointerId)) {
            switcher.releasePointerCapture(pointerId);
        }

        if (animate) {
            switcher.classList.remove('is-dragging');
            window.requestAnimationFrame(function () {
                thumb.style.removeProperty('transform');
            });
        } else {
            switcher.classList.remove('is-dragging');
            thumb.style.removeProperty('transform');
        }

        options[targetIndex].focus({ preventScroll: true });
        suppressClick = true;
        window.setTimeout(function () {
            suppressClick = false;
        }, 0);
    }

    function cancelDrag() {
        if (!dragState) return;

        var pointerId = dragState.pointerId;
        dragState = null;
        if (switcher.hasPointerCapture && switcher.hasPointerCapture(pointerId)) {
            switcher.releasePointerCapture(pointerId);
        }
        switcher.classList.remove('is-dragging');
        window.requestAnimationFrame(function () {
            thumb.style.removeProperty('transform');
        });
    }

    function initializeSwitcher() {
        switcher = document.getElementById('theme-switcher');
        if (!switcher) return;

        thumb = switcher.querySelector('.theme-switcher__thumb');
        options = Array.prototype.slice.call(switcher.querySelectorAll('.theme-switcher__option'));
        renderSwitcher();

        switcher.addEventListener('click', function (event) {
            var option = event.target.closest('.theme-switcher__option');
            if (!option || suppressClick) {
                event.preventDefault();
                return;
            }
            applyMode(option.getAttribute('data-theme-mode'), true);
        });

        switcher.addEventListener('keydown', function (event) {
            var option = event.target.closest('.theme-switcher__option');
            if (!option) return;

            var index = modeIndex(option.getAttribute('data-theme-mode'));
            if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
                index = Math.min(MODES.length - 1, index + 1);
            } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
                index = Math.max(0, index - 1);
            } else if (event.key === 'Home') {
                index = 0;
            } else if (event.key === 'End') {
                index = MODES.length - 1;
            } else {
                return;
            }

            event.preventDefault();
            applyMode(MODES[index], true);
            options[index].focus();
        });

        switcher.addEventListener('pointerdown', function (event) {
            if (!event.isPrimary || (event.pointerType === 'mouse' && event.button !== 0)) return;

            var geometry = getGeometry();
            var rect = switcher.getBoundingClientRect();
            var pressedIndex = clamp(
                Math.floor((event.clientX - rect.left - geometry.trackLeft) / geometry.segmentWidth),
                0,
                MODES.length - 1
            );

            dragState = {
                pointerId: event.pointerId,
                startX: event.clientX,
                startOffset: modeIndex(currentMode) * geometry.segmentWidth,
                offset: modeIndex(currentMode) * geometry.segmentWidth,
                segmentWidth: geometry.segmentWidth,
                maxOffset: geometry.maxOffset,
                pressedIndex: pressedIndex,
                moved: false
            };
            switcher.setPointerCapture(event.pointerId);
        });

        switcher.addEventListener('pointermove', function (event) {
            if (!dragState || event.pointerId !== dragState.pointerId) return;

            var delta = event.clientX - dragState.startX;
            if (!dragState.moved && Math.abs(delta) < 4) return;

            dragState.moved = true;
            dragState.offset = clamp(dragState.startOffset + delta, 0, dragState.maxOffset);
            switcher.classList.add('is-dragging');
            thumb.style.transform = 'translateX(' + dragState.offset + 'px)';
            event.preventDefault();
        });

        switcher.addEventListener('pointerup', function (event) {
            if (!dragState || event.pointerId !== dragState.pointerId) return;

            var targetIndex = dragState.moved
                ? Math.round(dragState.offset / dragState.segmentWidth)
                : dragState.pressedIndex;
            finishDrag(targetIndex, dragState.moved);
        });

        switcher.addEventListener('pointercancel', cancelDrag);
        switcher.addEventListener('lostpointercapture', function () {
            if (dragState) cancelDrag();
        });
    }

    function handleSystemPreferenceChange() {
        if (currentMode === 'system') {
            applyMode('system', false);
        }
    }

    applyMode(currentMode, false);

    if (systemPreference) {
        if (systemPreference.addEventListener) {
            systemPreference.addEventListener('change', handleSystemPreferenceChange);
        } else if (systemPreference.addListener) {
            systemPreference.addListener(handleSystemPreferenceChange);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeSwitcher);
    } else {
        initializeSwitcher();
    }
}());
