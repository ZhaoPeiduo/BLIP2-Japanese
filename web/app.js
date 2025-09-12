(() => {
  const { useState, useRef } = React;

  function App() {
    const [file, setFile] = useState(null);
    const [imgURL, setImgURL] = useState("");
    const [caption, setCaption] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [progress, setProgress] = useState(0);
    const fileRef = useRef();

    const onSelect = (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      setFile(f);
      setImgURL(URL.createObjectURL(f));
      setCaption("");
      setError("");
      setProgress(0);
    };

    const clearImage = () => {
      setFile(null);
      setImgURL("");
      setCaption("");
      setError("");
      setProgress(0);
      if (fileRef.current) fileRef.current.value = "";
    };

    const generate = async () => {
      if (!file) return;
      setLoading(true);
      setCaption("");
      setError("");
      setProgress(10);
      try {
        const form = new FormData();
        form.append("image", file);

        const res = await fetch("/api/caption", { method: "POST", body: form });
        setProgress(70);
        if (!res.ok) {
          const msg = await res.text();
          throw new Error(msg || `HTTP ${res.status}`);
        }
        const data = await res.json();
        setProgress(100);
        setCaption(data.caption || "");
      } catch (e) {
        setError(e?.message || String(e));
      } finally {
        setLoading(false);
        setTimeout(() => setProgress(0), 800);
      }
    };

    return (
      React.createElement('div', { className: 'card' },
        React.createElement('div', { className: 'header' },
          React.createElement('div', null,
            React.createElement('div', { className: 'title' }, 'Japanese Caption Generator'),
            React.createElement('div', { className: 'subtitle' }, 'BLIP-2 (Japanese) — Upload an image to caption')
          ),
          React.createElement('div', { className: 'upload' },
            React.createElement('input', { ref: fileRef, type: 'file', accept: 'image/*', onChange: onSelect })
          )
        ),
        React.createElement('div', { className: 'grid' },
          React.createElement('div', { className: 'panel' },
            React.createElement('h3', null, 'Preview'),
            React.createElement('div', { className: 'preview' },
              imgURL ? React.createElement('img', { src: imgURL, alt: 'preview' }) : 'No image selected'
            ),
            React.createElement('div', { className: 'actions' },
              React.createElement('button', { onClick: generate, disabled: !file || loading }, loading ? 'Generating…' : 'Generate Caption'),
              React.createElement('button', { className: 'secondary', onClick: clearImage, disabled: loading && !!file }, 'Clear')
            ),
            React.createElement('div', { className: 'progress', style: { marginTop: 8 } },
              React.createElement('div', { className: 'bar', style: { width: `${progress}%` } })
            )
          ),
          React.createElement('div', { className: 'panel' },
            React.createElement('h3', null, 'Caption'),
            error ? React.createElement('div', { className: 'error' }, error) : null,
            React.createElement('div', { className: 'output' }, caption || (loading ? 'Thinking…' : ''))
          )
        ),
        React.createElement('div', { className: 'footer' },
          React.createElement('div', { className: 'hint' }, 'Tip: Large images may take longer.'),
          React.createElement('div', { className: 'hint' }, 'Runs locally — no uploads to cloud')
        )
      )
    );
  }

  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(React.createElement(App));
})();

