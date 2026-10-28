document.addEventListener("submit", (event) => {
  const form = event.target;
  const pressed = event.submitter;
  for (const button of form.querySelectorAll("button")) {
    button.disabled = true;
  }
  if (pressed && pressed.dataset.busy) {
    if (pressed.name) {
      const field = document.createElement("input");
      field.type = "hidden";
      field.name = pressed.name;
      field.value = pressed.value;
      form.appendChild(field);
    }
    pressed.textContent = pressed.dataset.busy;
  }
});

window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    window.location.reload();
  }
});
