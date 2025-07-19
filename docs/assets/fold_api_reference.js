// Automatically fold the API reference section on page load
document.addEventListener("DOMContentLoaded", function() {
  const reference_link = document.querySelector('a[href$="reference/SUMMARY.html"]');
  const inp = reference_link.parentElement.parentElement.querySelector('input');
  inp.classList.remove('md-toggle--indeterminate');
  const nav = document.querySelector(`[aria-labelledby="${inp.id}_label"]`);
  nav.setAttribute('aria-expanded', 'false');
});