(function () {
    'use strict';

    var STORAGE_KEY = 'language';
    var LANGUAGES = ['en', 'zh-CN'];
    var switcher = null;
    var thumb = null;
    var options = [];
    var currentLanguage = document.documentElement.lang === 'zh-CN' ? 'zh-CN' : 'en';
    var dragState = null;
    var suppressClick = false;

    function storeLanguage(language) {
        try {
            window.localStorage.setItem(STORAGE_KEY, language);
        } catch (error) {
            // URL-based language selection still works when storage is unavailable.
        }
    }

    function languageIndex(language) {
        return Math.max(0, LANGUAGES.indexOf(language));
    }

    function isAvailable(index) {
        return Boolean(options[index] && options[index].getAttribute('data-language-url'));
    }

    function renderSwitcher() {
        if (!switcher) return;

        switcher.setAttribute('data-language', currentLanguage);
        options.forEach(function (option) {
            var isActive = option.getAttribute('data-language-code') === currentLanguage;
            option.setAttribute('aria-checked', isActive ? 'true' : 'false');
            option.setAttribute('tabindex', isActive ? '0' : '-1');
            option.classList.toggle('is-active', isActive);
        });
    }

    function selectLanguage(index) {
        if (!isAvailable(index)) {
            renderSwitcher();
            return;
        }

        var option = options[index];
        var language = option.getAttribute('data-language-code');
        var targetUrl = option.getAttribute('data-language-url');
        storeLanguage(language);

        if (language === currentLanguage) {
            renderSwitcher();
            option.focus({ preventScroll: true });
            return;
        }

        window.location.assign(targetUrl);
    }

    function getGeometry() {
        var switcherRect = switcher.getBoundingClientRect();
        var firstOptionRect = options[0].getBoundingClientRect();
        return {
            trackLeft: firstOptionRect.left - switcherRect.left,
            segmentWidth: firstOptionRect.width,
            maxOffset: firstOptionRect.width * (LANGUAGES.length - 1)
        };
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function endDrag(targetIndex) {
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

        if (isAvailable(targetIndex)) {
            selectLanguage(targetIndex);
        } else {
            renderSwitcher();
        }

        suppressClick = true;
        window.setTimeout(function () { suppressClick = false; }, 0);
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
        switcher = document.getElementById('language-switcher');
        if (!switcher) return;

        thumb = switcher.querySelector('.language-switcher__thumb');
        options = Array.prototype.slice.call(switcher.querySelectorAll('.language-switcher__option'));
        renderSwitcher();

        switcher.addEventListener('click', function (event) {
            var option = event.target.closest('.language-switcher__option');
            if (!option || suppressClick) {
                event.preventDefault();
                return;
            }
            selectLanguage(languageIndex(option.getAttribute('data-language-code')));
        });

        switcher.addEventListener('keydown', function (event) {
            var option = event.target.closest('.language-switcher__option');
            if (!option) return;

            var index = languageIndex(option.getAttribute('data-language-code'));
            if (event.key === 'ArrowRight' || event.key === 'ArrowDown' || event.key === 'End') {
                index = LANGUAGES.length - 1;
            } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp' || event.key === 'Home') {
                index = 0;
            } else {
                return;
            }

            event.preventDefault();
            if (isAvailable(index)) selectLanguage(index);
        });

        switcher.addEventListener('pointerdown', function (event) {
            if (!event.isPrimary || (event.pointerType === 'mouse' && event.button !== 0)) return;

            var geometry = getGeometry();
            var rect = switcher.getBoundingClientRect();
            var pressedIndex = clamp(
                Math.floor((event.clientX - rect.left - geometry.trackLeft) / geometry.segmentWidth),
                0,
                LANGUAGES.length - 1
            );

            dragState = {
                pointerId: event.pointerId,
                startX: event.clientX,
                startOffset: languageIndex(currentLanguage) * geometry.segmentWidth,
                offset: languageIndex(currentLanguage) * geometry.segmentWidth,
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
            endDrag(targetIndex);
        });

        switcher.addEventListener('pointercancel', cancelDrag);
        switcher.addEventListener('lostpointercapture', function () {
            if (dragState) cancelDrag();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeSwitcher);
    } else {
        initializeSwitcher();
    }
}());
