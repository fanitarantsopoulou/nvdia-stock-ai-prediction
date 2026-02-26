<script setup>
import { ref, onMounted, onUnmounted } from 'vue' // Προστέθηκε το onUnmounted
import axios from 'axios'
import { Line } from 'vue-chartjs'
import { Chart as ChartJS, Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement, Filler } from 'chart.js'

ChartJS.register(Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement, Filler)

const prediction = ref(null)
const chartData = ref(null)
const loading = ref(true)
const logs = ref([])

const addLog = (msg) => {
  logs.value.unshift(`[${new Date().toLocaleTimeString()}] ${msg}`)
  if (logs.value.length > 8) logs.value.pop()
}

const fetchData = async () => {
  loading.value = true;
  addLog("INITIALIZING NEURAL_CORE...");
  
  try {
    const res = await axios.get('http://127.0.0.1:8000/predict');
    
    // Έλεγχος αν τα δεδομένα είναι έγκυρα
    if (!res.data || !res.data.history_times || res.data.history_times.length === 0) {
      throw new Error("Empty data received from API");
    }

    prediction.value = res.data;
    addLog("FETCHING MARKET_DATA: SUCCESS");
    addLog(`SENTIMENT_SCORE: ${res.data.sentiment_score}`);
    addLog("RUNNING LSTM_INFERENCE...");

    // 1. Παίρνουμε τα έτοιμα strings ("HH:MM") απευθείας από το backend
    const historicalLabels = res.data.history_times;
    
    // 2. Υπολογισμός Forecast Label (προσθέτουμε 1 ώρα στο τελευταίο label)
    const lastTimeStr = historicalLabels[historicalLabels.length - 1]; 
    const [hours, minutes] = lastTimeStr.split(':').map(Number);
    const nextHour = (hours + 1) % 24;
    const forecastLabel = `${nextHour.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')} >> PRED`;

    // 3. Χτίσιμο του Chart Data
    chartData.value = {
      labels: [...historicalLabels, forecastLabel],
      datasets: [{
        label: 'PRICE_FLUX',
        // Δυναμικό Gradient ανάλογα με την κατεύθυνση (UP/DOWN)
        backgroundColor: (context) => {
          const chart = context.chart;
          const {ctx, chartArea} = chart;
          if (!chartArea) return null;
          const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
          gradient.addColorStop(0, res.data.direction === 'UP' ? 'rgba(0, 255, 101, 0.3)' : 'rgba(255, 0, 60, 0.3)');
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
          return gradient;
        },
        borderColor: res.data.direction === 'UP' ? '#00ff41' : '#ff003c',
        // Συνδυάζουμε το ιστορικό με την πρόβλεψη σε μια ενιαία γραμμή
        data: [...res.data.history, res.data.predicted_price],
        fill: true,
        tension: 0.3,
        borderWidth: 2,
        pointBackgroundColor: '#fff',
        // Εμφάνιση κύκλου ΜΟΝΟ στο σημείο της πρόβλεψης
        pointRadius: (context) => context.dataIndex === res.data.history.length ? 6 : 0,
        pointHoverRadius: 8,
        // Διακεκομμένη γραμμή (dashed) μόνο για το τελευταίο κομμάτι της πρόβλεψης
        segment: {
          borderDash: (context) => context.p1DataIndex === res.data.history.length ? [5, 5] : [],
        }
      }]
    };
    
    addLog("PREDICTION_READY: " + res.data.direction);
    
  } catch (e) {
    console.error("Critical API Error:", e);
    addLog("ERR: CONNECTION_TERMINATED");
    // Fallback labels για να μην σπάσει το UI
    chartData.value = null;
  } finally {
    loading.value = false;
  }
};

onMounted(() => {
  fetchData();
  // Auto-refresh interval
  const interval = setInterval(fetchData, 300000); 
  
  // Clean up interval on component destroy
  onUnmounted(() => {
    clearInterval(interval);
  });
})
</script>

<template>
  <div class="tech-bg"></div>
  <div class="scanlines"></div>

  <div class="relative z-10 min-h-screen text-[#00ff41] font-mono p-4 md:p-8">
    <div class="max-w-6xl mx-auto">
      
      <header class="flex justify-between items-end mb-8 border-b-2 border-[#00ff41] pb-2 bg-black bg-opacity-40 backdrop-blur-md px-4">
        <div>
          <h1 class="text-3xl font-black tracking-tighter uppercase italic">
            NVDA STOCK <span class="text-white">AI PREDICTION</span>
          </h1>
          <p class="text-[9px] opacity-50 tracking-widest uppercase mt-1">Nasdaq Real-Time Feed Active</p>
        </div>
        <div class="text-right">
          <div class="text-xs font-bold animate-pulse text-black bg-[#00ff41] px-2 mb-1 shadow-[0_0_10px_#00ff41]">SYSTEM_LIVE</div>
          <div class="text-[10px] opacity-60">* NASDAQ_TZ_SYNC_ENABLED</div>
        </div>
      </header>

      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        <div class="lg:col-span-1 bg-black bg-opacity-60 border border-[#00ff41] p-4 text-[10px] h-[450px] flex flex-col shadow-[inset_0_0_15px_rgba(0,255,65,0.1)]">
          <h3 class="border-b border-[#00ff41] mb-4 pb-1 font-bold tracking-widest text-white uppercase">Terminal_Logs</h3>
          <div class="flex-1 overflow-y-auto space-y-3 opacity-80 custom-scrollbar">
            <div v-for="(log, i) in logs" :key="i" :class="i === 0 ? 'text-white font-bold' : 'text-[#00ff41] opacity-50'">
              {{ log }}
            </div>
          </div>
          <div class="mt-4 pt-2 border-t border-[#00ff41] text-[9px] opacity-40">
            STABLE_DATALINK_ESTABLISHED
          </div>
        </div>

        <div class="lg:col-span-3 space-y-6">
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="border border-[#00ff41] p-4 bg-black bg-opacity-60 relative">
              <span class="absolute top-0 right-0 text-[8px] px-1 bg-[#00ff41] text-black font-bold uppercase">Live</span>
              <p class="text-[10px] opacity-50 uppercase font-bold">Current USD</p>
              <p class="text-3xl font-bold tracking-tighter text-white">${{ prediction?.current_price || '---' }}</p>
            </div>
            
            <div class="border border-[#00ff41] p-4 bg-black bg-opacity-60 relative">
              <span class="absolute top-0 right-0 text-[8px] px-1 bg-[#00ff41] text-black font-bold uppercase italic">AI_Inference</span>
              <p class="text-[10px] opacity-50 uppercase italic font-bold">1H_Forecast</p>
              <p :class="prediction?.direction === 'UP' ? 'text-[#00ff41]' : 'text-[#ff003c]'" class="text-3xl font-bold tracking-tighter">
                ${{ prediction?.predicted_price || '---' }}
              </p>
            </div>

            <div class="border border-[#00ff41] p-4 bg-black bg-opacity-60 relative">
              <span class="absolute top-0 right-0 text-[8px] px-1 bg-blue-500 text-white font-bold uppercase font-mono">FinBert</span>
              <p class="text-[10px] opacity-50 uppercase font-bold">Sentiment_Index</p>
              <p class="text-3xl font-bold text-blue-400 tracking-tighter">{{ prediction?.sentiment_score ?? '---' }}</p>
            </div>
          </div>

          <div class="border border-[#00ff41] p-6 bg-black bg-opacity-60 relative h-[360px] shadow-[0_0_30px_rgba(0,255,65,0.05)]">
            <div class="flex justify-between items-center mb-6">
               <span class="text-xs font-bold tracking-[0.3em] uppercase">Market_Visualizer_V1</span>
               <button @click="fetchData" class="border border-[#00ff41] text-[10px] px-4 py-1 hover:bg-[#00ff41] hover:text-black transition-all duration-300 font-bold active:scale-95">
                 RUN_ANALYSIS()
               </button>
            </div>
            
            <div v-if="loading" class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-90 z-20 border border-[#00ff41]">
              <p class="animate-pulse font-bold tracking-[0.5em] text-white uppercase">Recalibrating_Neural_Weights...</p>
            </div>

            <div class="h-[260px]">
              <Line v-if="chartData" :data="chartData" :options="{ 
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: { grid: { color: 'rgba(0, 255, 65, 0.05)' }, ticks: { color: 'rgba(0, 255, 65, 0.7)', font: { size: 10, family: 'monospace' } } },
                  x: { grid: { display: false }, ticks: { color: 'rgba(0, 255, 65, 0.7)', font: { size: 10, family: 'monospace' } } }
                },
                plugins: { legend: { display: false }, tooltip: { backgroundColor: '#000', borderColor: '#00ff41', borderWidth: 1, titleFont: { family: 'monospace' }, bodyFont: { family: 'monospace' } } }
              }" />
            </div>
          </div>
        </div>
      </div>

      <footer class="mt-8 text-center text-[9px] opacity-30 uppercase tracking-[0.5em]">
        Neural Architecture v1.0 // No Data Leak Detected // NVIDIA_QUANTA_CORE
      </footer>
    </div>
  </div>
</template>

<style scoped>
/* Scrollbar Styling for Logs */
.custom-scrollbar::-webkit-scrollbar {
  width: 4px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.3);
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: #00ff41;
}

/* Background & Scanlines logic remains same as per global style */
</style>