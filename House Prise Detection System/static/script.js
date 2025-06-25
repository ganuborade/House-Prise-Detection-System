document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(e.target);
  const jsonData = {};
  formData.forEach((value, key) => {
    jsonData[key] = parseFloat(value);
  });

  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(jsonData)
  });

  const result = await response.json();
  document.getElementById("result").innerHTML = `
    Predicted House Price: ₹${result.price_inr.toLocaleString()} <br><br>
    <img src="data:image/png;base64,${result.graph}" style="max-width: 100%; border: 1px solid #ccc; border-radius: 8px;">
  `;
});
