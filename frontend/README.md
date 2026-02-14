# Frontend Documentation

## Structure

```
frontend/
├── index.html          # Main HTML file
├── css/
│   └── styles.css      # All styling
└── js/
    ├── api.js          # API communication & utilities
    └── app.js          # Main application logic
```

## Files Explanation

### `index.html`
Clean HTML structure with:
- Upload section for CSV files
- Analysis configuration (protected attributes, label column, etc.)
- Results display area
- Mitigation controls with download option

### `css/styles.css`
All styling in one place:
- Modern, clean design
- Responsive layout
- Button states and colors
- Message boxes (info, success, error)

### `js/api.js`
API communication layer:
- `API.uploadDataset()` - Upload CSV file
- `API.analyze()` - Run bias analysis
- `API.mitigate()` - Apply mitigation
- `Utils` - Helper functions for messages and suggestions

### `js/app.js`
Main application logic:
- `BiasDetectionApp` class manages the entire workflow
- Event handlers for all buttons
- UI state management
- Results display

## How to Use

### Local Development
1. Start Flask backend:
   ```bash
   python flask_app.py
   ```

2. Open `index.html` in a browser (you can use Python's built-in server):
   ```bash
   cd frontend
   python -m http.server 8000
   ```
   Then visit: http://localhost:8000

### Deploy on GitHub Pages
1. Push the `frontend/` folder to your GitHub repository
2. Enable GitHub Pages in repository settings
3. Set source to the branch containing frontend files
4. Access via: `https://your-username.github.io/your-repo/frontend/`
5. Add `?api=https://your-backend-url` to connect to your deployed backend

### Using a Different Backend
Add the API URL as a query parameter:
```
https://your-frontend.com/?api=https://your-backend.com
```

## Features

✅ Clean separation of concerns (HTML, CSS, JS)  
✅ No build process required - works with plain files  
✅ Can be deployed as static site  
✅ Works with Flask backend on different domain (CORS enabled)  
✅ Automatic mitigation suggestions based on analysis  
✅ Download mitigated datasets  
✅ Modern, responsive UI  

## API Endpoints Used

- `POST /upload_dataset` - Upload CSV file
- `POST /analyze` - Run bias analysis
- `POST /mitigate` - Apply mitigation
- `GET /download_dataset` - Download mitigated CSV
