(function () {
    'use strict';

    function initializeBlogFilter() {
        var tabs = Array.prototype.slice.call(document.querySelectorAll('.blog-category-tab'));
        var cards = Array.prototype.slice.call(document.querySelectorAll('.blog-card[data-category]'));
        var yearGroups = Array.prototype.slice.call(document.querySelectorAll('.blog-year-group'));
        var yearLinks = Array.prototype.slice.call(document.querySelectorAll('.blog-year-link'));
        var searchInput = document.getElementById('blog-search-input');
        var clearButton = document.getElementById('blog-search-clear');
        var status = document.getElementById('blog-search-status');
        var emptyState = document.getElementById('blog-empty-state');
        var activeCategory = 'all';
        var query = '';

        if (!searchInput || !clearButton || !status || !emptyState) return;

        function normalize(value) {
            var normalized = String(value || '');
            if (typeof normalized.normalize === 'function') {
                normalized = normalized.normalize('NFKC');
            }
            return normalized.toLocaleLowerCase().trim().replace(/\s+/g, ' ');
        }

        function getSearchTerms(value) {
            var normalized = normalize(value);
            return normalized ? normalized.split(' ') : [];
        }

        function updateYearGroup(group) {
            var groupCards = Array.prototype.slice.call(group.querySelectorAll('.blog-card'));
            var visibleCards = groupCards.filter(function (card) { return !card.hidden; });

            group.hidden = visibleCards.length === 0;
            groupCards.forEach(function (card) { card.classList.remove('border-0'); });
            if (visibleCards.length > 0) {
                visibleCards[visibleCards.length - 1].classList.add('border-0');
            }
        }

        function updateStatus(visibleCount) {
            var template = visibleCount === 1
                ? status.getAttribute('data-count-one')
                : status.getAttribute('data-count-many');
            status.textContent = template.replace('%s', visibleCount);
        }

        function applyFilters() {
            var searchTerms = getSearchTerms(query);
            var visibleCount = 0;

            cards.forEach(function (card) {
                var categoryMatches = activeCategory === 'all' || card.getAttribute('data-category') === activeCategory;
                var searchableText = normalize(card.getAttribute('data-search-text'));
                var searchMatches = searchTerms.every(function (term) {
                    return searchableText.indexOf(term) !== -1;
                });
                var isVisible = categoryMatches && searchMatches;

                card.hidden = !isVisible;
                if (isVisible) visibleCount += 1;
            });

            yearGroups.forEach(updateYearGroup);
            yearLinks.forEach(function (link) {
                var year = link.getAttribute('data-year');
                var group = document.querySelector('.blog-year-group[data-year="' + year + '"]');
                link.hidden = !group || group.hidden;
            });

            clearButton.hidden = query.length === 0;
            emptyState.hidden = visibleCount !== 0;
            updateStatus(visibleCount);
        }

        tabs.forEach(function (tab) {
            tab.addEventListener('click', function () {
                activeCategory = tab.getAttribute('data-category');
                tabs.forEach(function (candidate) {
                    candidate.classList.toggle('active', candidate === tab);
                });
                applyFilters();
            });
        });

        searchInput.addEventListener('input', function () {
            query = searchInput.value;
            applyFilters();
        });

        searchInput.addEventListener('keydown', function (event) {
            if (event.key === 'Escape' && query) {
                searchInput.value = '';
                query = '';
                applyFilters();
            }
        });

        clearButton.addEventListener('click', function () {
            searchInput.value = '';
            query = '';
            applyFilters();
            searchInput.focus();
        });

        applyFilters();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeBlogFilter);
    } else {
        initializeBlogFilter();
    }
}());
