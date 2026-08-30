"""
Script to run the FastAPI backend server.
Run this script to start the backend API server.
"""
import uvicorn

if __name__ == "__main__":
    print("="*80)
    print("🚀 Starting Medi-Guard Backend Server")
    print("="*80)
    print("\n📡 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔌 Prediction Endpoint: http://localhost:8000/predict")
    print("\n" + "="*80)
    print("Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

