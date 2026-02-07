document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("textForm");
  const resultDiv = document.getElementById("result");

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = document.getElementById("textInput").value.trim();
      if (!text) return;

      resultDiv.classList.add("d-none");

      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: [text] }),
      });
      const data = await res.json();
      const sentiment = data.predictions[0].sentiment;

      resultDiv.textContent = `Sentiment: ${sentiment}`;
      resultDiv.className = "alert alert-info mt-3";
      resultDiv.classList.remove("d-none");
    });
  }

  const uploadForm = document.getElementById("uploadForm");
  if (uploadForm) {
    uploadForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById("fileInput");
      const file = fileInput.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/upload_csv", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      let html = `<h5>Predictions:</h5><table class="table table-bordered mt-3"><tr><th>Text</th><th>Sentiment</th></tr>`;
      data.predictions.forEach(p => {
        html += `<tr><td>${p.text}</td><td>${p.sentiment}</td></tr>`;
      });
      html += "</table>";
      document.getElementById("csvResults").innerHTML = html;
    });
  }
});

// ===== LIGHTBOX FUNCTIONALITY =====
document.addEventListener("DOMContentLoaded", () => {
  const galleryImages = document.querySelectorAll(".gallery-img");
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.querySelector(".lightbox-img");
  const lightboxCaption = document.querySelector(".lightbox-caption");
  const closeBtn = document.querySelector(".close-btn");
  const prevBtn = document.querySelector(".prev");
  const nextBtn = document.querySelector(".next");

  let currentIndex = 0;

  const openLightbox = (index) => {
    currentIndex = index;
    const image = galleryImages[index];
    lightboxImg.src = image.src;
    lightboxCaption.textContent = image.getAttribute("data-caption");
    lightbox.style.display = "flex";
  };

  const closeLightbox = () => (lightbox.style.display = "none");

  const showNext = () => {
    currentIndex = (currentIndex + 1) % galleryImages.length;
    openLightbox(currentIndex);
  };

  const showPrev = () => {
    currentIndex = (currentIndex - 1 + galleryImages.length) % galleryImages.length;
    openLightbox(currentIndex);
  };

  galleryImages.forEach((img, index) => {
    img.addEventListener("click", () => openLightbox(index));
  });

  closeBtn.addEventListener("click", closeLightbox);
  nextBtn.addEventListener("click", showNext);
  prevBtn.addEventListener("click", showPrev);

  // Close when clicking outside
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) closeLightbox();
  });

  // Keyboard support
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") showNext();
    if (e.key === "ArrowLeft") showPrev();
    if (e.key === "Escape") closeLightbox();
  });
});
