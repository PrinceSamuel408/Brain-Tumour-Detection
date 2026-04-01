import { motion, AnimatePresence } from "framer-motion";
import { Upload, Brain, Scan } from "lucide-react";

interface DropZoneOverlayProps {
  isDragOver: boolean;
  isScanning: boolean;
  scanProgress: number;
  hasResults: boolean;
  onFileSelect: (file: File) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  onDrop: (e: React.DragEvent) => void;
}

const DropZoneOverlay = ({
  isDragOver,
  isScanning,
  scanProgress,
  hasResults,
  onFileSelect,
  onDragOver,
  onDragLeave,
  onDrop,
}: DropZoneOverlayProps) => {
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) onFileSelect(e.target.files[0]);
  };

  return (
    <div className="absolute inset-0 pointer-events-none z-10">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.3 }}
        className="absolute top-8 left-1/2 -translate-x-1/2 text-center pointer-events-none"
      >
        <div className="flex items-center gap-3 mb-2">
          <Brain className="w-6 h-6 text-primary" />
          <h1 className="font-display text-2xl tracking-wider glow-text-primary">
            NEURO<span className="text-foreground">SCAN</span>
          </h1>
        </div>
        <p className="text-muted-foreground text-sm font-body tracking-wide">
          AI-Powered Brain Tumor Detection System
        </p>
      </motion.div>

      {/* Drop zone area — only show when no results and not scanning */}
      <AnimatePresence>
        {!hasResults && !isScanning && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.5 }}
            className="absolute bottom-12 left-1/2 -translate-x-1/2 pointer-events-auto"
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <label
              className={`
                glass-panel rounded-2xl px-10 py-8 flex flex-col items-center gap-4 cursor-pointer
                transition-all duration-500 border
                ${isDragOver ? "glow-border-active" : "glow-border-primary"}
              `}
            >
              <motion.div
                animate={isDragOver ? { scale: 1.2, rotate: 180 } : { scale: 1, rotate: 0 }}
                transition={{ type: "spring", stiffness: 200 }}
              >
                <Upload className="w-8 h-8 text-primary" />
              </motion.div>
              <div className="text-center">
                <p className="font-display text-sm tracking-wider text-foreground">
                  {isDragOver ? "RELEASE TO SCAN" : "DROP MRI SCAN"}
                </p>
                <p className="text-muted-foreground text-xs mt-1 font-body">
                  or click to browse • DICOM, PNG, JPG
                </p>
              </div>
              <input
                type="file"
                className="hidden"
                accept="image/*,.dcm"
                onChange={handleFileInput}
              />
            </label>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Scanning status */}
      <AnimatePresence>
        {isScanning && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="absolute bottom-12 left-1/2 -translate-x-1/2 pointer-events-none"
          >
            <div className="glass-panel rounded-2xl px-10 py-6 flex flex-col items-center gap-3">
              <Scan className="w-6 h-6 text-primary animate-pulse-glow" />
              <p className="font-display text-sm tracking-wider glow-text-primary">
                ANALYZING NEURAL TISSUE
              </p>
              <div className="w-48 h-1 rounded-full bg-muted overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-primary"
                  initial={{ width: "0%" }}
                  animate={{ width: `${scanProgress * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
              <p className="text-muted-foreground text-xs font-body">
                {Math.round(scanProgress * 100)}% — Running inference model...
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DropZoneOverlay;
