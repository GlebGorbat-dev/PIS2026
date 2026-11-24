document.addEventListener("change", (event) => {
  const input = event.target;
  if (!(input instanceof HTMLInputElement) || input.type !== "file") return;
  const slot = input.form?.querySelector(".file-chosen");
  if (!slot) return;
  slot.textContent = input.files?.[0]?.name || slot.dataset.empty || "";
});

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
