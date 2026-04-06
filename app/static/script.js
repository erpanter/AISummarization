const fileInput = document.getElementById('file');
const drop = document.getElementById('drop');
const btn = document.getElementById('btn');
const prog = document.getElementById('prog');
const statusEl = document.getElementById('status');
const finalEl = document.getElementById('final');
const chunksEl = document.getElementById('chunks');
const downloadBtn = document.getElementById('download');
const metaEl = document.getElementById('meta');
const skeletonFinal = document.getElementById('skeleton-final');
const fileName = document.getElementById('file-name');

function setStatus(msg){ statusEl.textContent = msg || ""; }
function setBusy(b){
  btn.disabled = b;
  prog.style.display = b ? 'inline-block' : 'none';
  if (!b) prog.value = 0;
}
function setFile(file){
  metaEl.style.display = file ? 'inline-block' : 'none';
  metaEl.textContent = file ? `${file.name} • ${(file.size/1024/1024).toFixed(2)} MB` : "";
}

 drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
 drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
 drop.addEventListener('drop', e => {
   e.preventDefault(); drop.classList.remove('drag');
   const f = e.dataTransfer.files[0];
   if (f) { fileInput.files = e.dataTransfer.files; setFile(f); }
 });

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];

  if (file) {
    fileName.textContent = file.name;
    fileName.style.display = "block";
  } else {
    fileName.style.display = "none";
  }
});

btn.addEventListener('click', async ()=>{
  const file = fileInput.files[0];
  if (!file) return alert("Pick a file first 😭");

  finalEl.textContent = "";
  chunksEl.innerHTML = "";
  downloadBtn.style.display = "none";

  skeletonFinal.style.display = 'block';
  finalEl.style.display = 'none';

  setBusy(true);
  setStatus("Processing...");

  const fd = new FormData(); fd.append("file", file);

  try {
    const res = await fetch("/api/summarize", { method: "POST", body: fd });
    if (!res.ok) throw new Error("Failed request");

    const data = await res.json();

    skeletonFinal.style.display = 'none';
    finalEl.style.display = 'block';

    finalEl.textContent = data.final_summary || "(empty)";

    (data.chunks || []).forEach(c => {
      const li = document.createElement('li');
      li.textContent = c;
      chunksEl.appendChild(li);
    });

    downloadBtn.style.display = "inline-block";
    downloadBtn.onclick = () => {
      const blob = new Blob([data.final_summary || ""], {type:"text/plain"});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name + "_summary.txt";
      a.click();
    };

    setStatus("Done ✔");

  } catch (e) {
    skeletonFinal.style.display = 'none';
    alert(e.message);
    setStatus("Error");
  } finally {
    setBusy(false);
  }
});