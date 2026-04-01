import { motion } from "framer-motion";
import { Activity, Cpu } from "lucide-react";

interface BrainSceneProps {
  isScanning: boolean;
  isDragOver: boolean;
  scanProgress: number;
}

const BrainScene = ({ isScanning, isDragOver, scanProgress }: BrainSceneProps) => {
  return (
    <div className="absolute inset-0 flex items-center justify-center overflow-hidden pointer-events-none opacity-20">
      {/* Background Gradient */}
      <div 
        className={`absolute inset-0 transition-opacity duration-1000 ${
          isDragOver ? "bg-primary/20" : "bg-transparent"
        }`} 
      />

      {/* Grid Pattern */}
      <div 
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: "linear-gradient(#333 1px, transparent 1px), linear-gradient(90deg, #333 1px, transparent 1px)",
          backgroundSize: "40px 40px"
        }}
      />

      {/* Animated Hexagons / Brain Abstract */}
      <motion.div 
        animate={{ 
          scale: isScanning ? 1.1 : 1,
          rotate: isScanning ? 5 : 0
        }}
        transition={{ duration: 4, repeat: isScanning ? Infinity : 0, repeatType: "reverse" }}
        className="relative flex items-center justify-center"
      >
        <div className="absolute w-[600px] h-[600px] rounded-full border border-primary/20 blur-[2px]" />
        <div className="absolute w-[400px] h-[400px] rounded-full border border-primary/30" />
        
        {isScanning && (
          <>
            <motion.div 
              className="absolute inset-0 flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 1, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Activity className="w-64 h-64 text-primary opacity-30" />
            </motion.div>
            
            {/* Scanning Laser Line */}
            <motion.div 
              className="absolute w-full h-[2px] bg-primary shadow-[0_0_15px_rgba(102,126,234,0.8)]"
              initial={{ top: "0%" }}
              animate={{ top: "100%" }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
            />
          </>
        )}

        {!isScanning && <Cpu className="w-32 h-32 text-muted-foreground opacity-20" />}
      </motion.div>
    </div>
  );
};

export default BrainScene;
