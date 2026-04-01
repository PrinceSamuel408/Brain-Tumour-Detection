import { motion } from "framer-motion";
import { Activity, XCircle, FileImage, ShieldCheck } from "lucide-react";

interface DetectionResult {
  label: string;
  confidence: number;
  isDetected: boolean;
  overlayImage?: string; // Base64 Grad-CAM string
}

interface ResultsPanelProps {
  results: DetectionResult[];
  onReset: () => void;
}

const ResultsPanel = ({ results, onReset }: ResultsPanelProps) => {
  // Assume the highest confidence result is first (the model's top prediction)
  const topMatch = results[0];

  return (
    <motion.div
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 50 }}
      transition={{ type: "spring", stiffness: 100, damping: 20 }}
      className="absolute right-0 top-0 bottom-0 w-[500px] glass-panel border-l 
                 z-20 flex flex-col p-8 overflow-y-auto"
    >
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="font-display text-2xl glow-text-primary">DIAGNOSIS REPORT</h2>
          <p className="text-muted-foreground text-sm font-body">
            AI Neuro-analysis completed
          </p>
        </div>
        <button
          onClick={onReset}
          className="p-2 mr-2 rounded-full hover:bg-white/5 transition-colors"
          title="Reset and start new scan"
        >
          <XCircle className="w-6 h-6 text-muted-foreground hover:text-white" />
        </button>
      </div>

      {/* Primary Result Card */}
      <div className="bg-primary/10 border border-primary/30 rounded-2xl p-6 mb-8 relative overflow-hidden">
        <div className="absolute -right-4 -top-4 opacity-5">
          <ShieldCheck className="w-48 h-48" />
        </div>
        <p className="text-sm font-display text-primary uppercase tracking-widest mb-2">
          Primary Finding
        </p>
        <h3 className="text-4xl font-display text-white mb-2">
          {topMatch.label.replace("notumor", "No Tumor")}
        </h3>
        <div className="flex items-center gap-3">
          <div className="h-2 flex-grow bg-white/10 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${topMatch.confidence}%` }}
              transition={{ delay: 0.5, duration: 1 }}
              className={`h-full rounded-full ${
                topMatch.isDetected ? "bg-red-500" : "bg-green-500"
              }`}
            />
          </div>
          <span className="font-mono text-xl">{topMatch.confidence.toFixed(1)}%</span>
        </div>
      </div>

      {/* Grad-CAM Overlay */}
      {topMatch.overlayImage && (
        <div className="mb-8">
          <h4 className="font-display text-sm text-muted-foreground mb-4 uppercase tracking-widest flex items-center gap-2">
            <FileImage className="w-4 h-4" /> Explanation (Grad-CAM)
          </h4>
          <div className="rounded-xl overflow-hidden border border-white/10 relative">
            <img src={topMatch.overlayImage} alt="Grad-CAM Overlay" className="w-full h-auto object-cover" />
            <div className="absolute bottom-2 right-2 bg-black/60 px-3 py-1 rounded-full text-xs font-mono text-white/80">
              Layer4[-1] focus
            </div>
          </div>
        </div>
      )}

      {/* Confidence Breakdown List */}
      <div className="flex-1">
        <h4 className="font-display text-sm text-muted-foreground mb-4 uppercase tracking-widest flex items-center gap-2">
          <Activity className="w-4 h-4" /> Confidence Breakdown
        </h4>
        <div className="space-y-4">
          {results.map((result, i) => (
            <motion.div
              key={result.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + i * 0.1 }}
              className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/5"
            >
              <span className="font-body text-white/90">
                {result.label.replace("notumor", "No Tumor")}
              </span>
              <div className="flex items-center gap-3">
                <span className="font-mono text-sm text-muted-foreground">
                  {result.confidence.toFixed(1)}%
                </span>
                <div 
                  className={`w-2 h-2 rounded-full ${
                    result.confidence > 50 ? "bg-primary" : "bg-white/20"
                  }`} 
                />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

export default ResultsPanel;
