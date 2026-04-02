// ========================================
// Nano-Claw-Code Website Interactions
// ========================================

document.addEventListener('DOMContentLoaded', () => {

  // --- Theme toggle (dark/light) ---
  const themeToggle = document.getElementById('themeToggle');
  const root = document.documentElement;

  // Restore saved preference, default to dark
  const savedTheme = localStorage.getItem('theme') || 'dark';
  root.setAttribute('data-theme', savedTheme);

  themeToggle.addEventListener('click', () => {
    const current = root.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  });

  // --- Scroll-triggered fade-in animations ---
  const observerOptions = {
    threshold: 0.15,
    rootMargin: '0px 0px -40px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Mark elements for fade-in
  const fadeTargets = [
    '.stat-card',
    '.about-text',
    '.code-window',
    '.results-table-wrapper',
    '.comparison',
    '.pipeline-stage',
    '.pipeline-arrow',
    '.arch-card',
    '.qs-step',
    '.roadmap-item'
  ];

  fadeTargets.forEach(selector => {
    document.querySelectorAll(selector).forEach(el => {
      el.classList.add('fade-in');
      observer.observe(el);
    });
  });

  // --- Staggered animations for grouped elements ---
  const staggerGroups = [
    { selector: '.hero-stats .stat-card', delay: 100 },
    { selector: '.quickstart-grid .qs-step', delay: 150 },
    { selector: '.roadmap-item', delay: 80 }
  ];

  staggerGroups.forEach(group => {
    const elements = document.querySelectorAll(group.selector);
    elements.forEach((el, i) => {
      el.style.transitionDelay = `${i * group.delay}ms`;
    });
  });

  // --- Smooth nav background on scroll ---
  const nav = document.querySelector('.nav');
  let lastScroll = 0;

  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;

    if (scrollY > 100) {
      nav.style.borderBottomColor = 'var(--border-light)';
    } else {
      nav.style.borderBottomColor = 'var(--border)';
    }

    lastScroll = scrollY;
  }, { passive: true });

  // --- Animated counter for stat numbers ---
  const animateCounter = (element, target, suffix = '') => {
    const duration = 1500;
    const start = performance.now();
    const isFloat = target % 1 !== 0;

    const step = (now) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = eased * target;

      if (isFloat) {
        element.childNodes[0].textContent = current.toFixed(1);
      } else {
        element.childNodes[0].textContent = Math.round(current);
      }

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    };

    requestAnimationFrame(step);
  };

  // Observe stat cards for counter animation
  const statObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const numberEl = entry.target.querySelector('.stat-number');
        if (numberEl && !numberEl.dataset.animated) {
          numberEl.dataset.animated = 'true';
          const text = numberEl.childNodes[0].textContent.trim();
          const value = parseFloat(text);
          if (!isNaN(value)) {
            animateCounter(numberEl, value);
          }
        }
        statObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  document.querySelectorAll('.stat-card').forEach(card => {
    statObserver.observe(card);
  });

  // --- Comparison bar animation ---
  const barObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const bars = entry.target.querySelectorAll('.bar');
        bars.forEach(bar => {
          const targetWidth = bar.style.width;
          bar.style.width = '0%';
          bar.style.transition = 'width 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              bar.style.width = targetWidth;
            });
          });
        });
        barObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });

  const comparison = document.querySelector('.comparison');
  if (comparison) {
    barObserver.observe(comparison);
  }
});
