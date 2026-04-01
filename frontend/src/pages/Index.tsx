import { useState, useCallback, useRef, useEffect } from "react";
import { AnimatePresence } from "framer-motion";
import BrainScene from "../components/BrainScene";
import DropZoneOverlay from "../components/DropZoneOverlay";
import ResultsPanel from "../components/ResultsPanel";

interface DetectionResult {
  label: string;
  confidence: number;
  isDetected: boolean;
  overlayImage?: string;
}

const Index = () => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [results, setResults] = useState<DetectionResult[] | null>(null);
  
  // Fake progress bar interval when waiting for API
  const progressRef = useRef<number | null>(null);

  const startFakeProgress = useCallback(() => {
    setScanProgress(0);
    let progress = 0;
    progressRef.current = window.setInterval(() => {
      // Go up to 90% and hold there until API returns
      if (progress < 0.9) {
        progress += 0.02 + Math.random() * 0.03;
        if (progress > 0.9) progress = 0.9;
        setScanProgress(progress);
      }
    }, 100);
  }, []);

  const stopFakeProgress = useCallback(() => {
    if (progressRef.current) clearInterval(progressRef.current);
    setScanProgress(1); // Jump to 100%
  }, []);

  const analyzeFile = async (file: File) => {
    setIsScanning(true);
    setResults(null);
    startFakeProgress();

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("API Request Failed");

      const data = await response.json();
      
      // Convert FastAPI response to Frontend format
      const isTumor = data.predicted_class !== "notumor";
      
      const convertedResults: DetectionResult[] = Object.entries(data.confidences)
        .sort((a: any, b: any) => b[1] - a[1]) // Sort highest confidence first
        .map(([className, score]: [string, any]) => ({
          label: className,
          confidence: score as number,
          isDetected: className === data.predicted_class && isTumor,
          // Attach the base64 overlay image only to the top prediction
          overlayImage: className === data.predicted_class ? data.overlay_image : undefined
        }));

      stopFakeProgress();
      setTimeout(() => {
        setIsScanning(false);
        setResults(convertedResults);
      }, 400); // Small delay for the 100% animation to finish

    } catch (error) {
      console.error(error);
      stopFakeProgress();
      setIsScanning(false);
      alert("Failed to connect to the NeuroScan backend. Is the Python server running?");
    }
  };

  useEffect(() => {
    return () => stopFakeProgress();
  }, [stopFakeProgress]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      analyzeFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (file: File) => {
    analyzeFile(file);
  };

  const handleReset = () => {
    setResults(null);
    setIsScanning(false);
    setScanProgress(0);
  };

  return (
    <div
      className="relative w-screen h-screen overflow-hidden bg-background"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <BrainScene
        isScanning={isScanning}
        isDragOver={isDragOver}
        scanProgress={scanProgress}
      />

      <DropZoneOverlay
        isDragOver={isDragOver}
        isScanning={isScanning}
        scanProgress={scanProgress}
        hasResults={!!results}
        onFileSelect={handleFileSelect}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      />

      <AnimatePresence>
        {results && <ResultsPanel results={results} onReset={handleReset} />}
      </AnimatePresence>
    </div>
  );
};

export default Index;
