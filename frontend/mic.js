document.getElementById("start").onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  console.log("Mic OK", stream);
};
