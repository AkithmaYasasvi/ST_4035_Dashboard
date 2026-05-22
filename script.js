// State Variables
let sampleSize = 1000;
let imbalanceRatio = 0.15; // Minority class ratio (e.g., 0.15 for 15%)
let balancingMethod = 'none'; // 'none', 'oversampling', 'undersampling', 'smote'
let linkFunction = 'logit'; // 'logit', 'probit', 'cloglog'

// Datasets
let originalDataset = [];
let balancedDataset = [];

// Model State
let fittedBeta = [0.0, 0.0, 0.0];
let aucValue = 0.5;

// Chart Instances
let classDistChart = null;
let fittedCurvesChart = null;
let rocChart = null;

// ==========================================
// Seeded Random Number Generator (Mulberry32)
// ==========================================
function createRandom(seed) {
    return function() {
        let t = seed += 0x6D2B79F5;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Box-Muller transform for normal distribution
function randomNormal(mean, stdDev, rand) {
    let u1 = rand();
    let u2 = rand();
    if (u1 < 1e-9) u1 = 1e-9; // Avoid log(0)
    let randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
    return mean + stdDev * randStdNormal;
}

// Standard Normal CDF (Probit Link Inverse)
// Abramowitz and Stegun polynomial approximation (formula 7.1.26)
function erf(x) {
    let sign = (x >= 0) ? 1 : -1;
    x = Math.abs(x);
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
}

function normalCDF(x) {
    return 0.5 * (1.0 + erf(x / Math.sqrt(2.0)));
}

function normalPDF(x) {
    return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

// ==========================================
// Data Generation & Balancing Pipelines
// ==========================================
function generateOriginalData() {
    let rand = createRandom(42); // Always use the same seed for consistent slider behavior
    let n1 = Math.floor(sampleSize * imbalanceRatio);
    let n0 = sampleSize - n1;
    
    originalDataset = [];
    
    // Class 0: Healthy Patients (No Risk)
    for (let i = 0; i < n0; i++) {
        let age = randomNormal(50, 10, rand);
        let sbp = randomNormal(122, 12, rand);
        // Ensure plausible bounds
        age = Math.max(18, Math.min(95, age));
        sbp = Math.max(90, Math.min(200, sbp));
        
        originalDataset.push({
            y: 0,
            age: age,
            sbp: sbp,
            // Standardize features: Mean 50 SD 10 for Age, Mean 130 SD 15 for SBP
            features: [1.0, (age - 50.0) / 10.0, (sbp - 130.0) / 15.0]
        });
    }
    
    // Class 1: High Risk Patients
    for (let i = 0; i < n1; i++) {
        let age = randomNormal(62, 8, rand);
        let sbp = randomNormal(148, 15, rand);
        // Ensure plausible bounds
        age = Math.max(18, Math.min(95, age));
        sbp = Math.max(90, Math.min(200, sbp));
        
        originalDataset.push({
            y: 1,
            age: age,
            sbp: sbp,
            // Standardize features in the same way
            features: [1.0, (age - 50.0) / 10.0, (sbp - 130.0) / 15.0]
        });
    }
}

function balanceData() {
    let class0 = originalDataset.filter(d => d.y === 0);
    let class1 = originalDataset.filter(d => d.y === 1);
    
    let rand = createRandom(100); // Separate seed for sampling stability
    
    if (balancingMethod === 'none') {
        balancedDataset = [...originalDataset];
    } 
    else if (balancingMethod === 'oversampling') {
        // Oversample minority (Class 1) to match majority (Class 0)
        let balancedClass1 = [...class1];
        if (class1.length > 0) {
            while (balancedClass1.length < class0.length) {
                let idx = Math.floor(rand() * class1.length);
                balancedClass1.push({ ...class1[idx] });
            }
        }
        balancedDataset = [...class0, ...balancedClass1];
    } 
    else if (balancingMethod === 'undersampling') {
        // Undersample majority (Class 0) to match minority (Class 1)
        let balancedClass0 = [];
        if (class0.length > 0 && class1.length > 0) {
            let tempClass0 = [...class0];
            // Shuffle majority
            for (let i = tempClass0.length - 1; i > 0; i--) {
                let j = Math.floor(rand() * (i + 1));
                [tempClass0[i], tempClass0[j]] = [tempClass0[j], tempClass0[i]];
            }
            balancedClass0 = tempClass0.slice(0, class1.length);
        }
        balancedDataset = [...balancedClass0, ...class1];
    } 
    else if (balancingMethod === 'smote') {
        // SMOTE: Synthesize minority samples to reach 85% of majority count
        let balancedClass1 = [...class1];
        let targetCount = Math.floor(class0.length * 0.85);
        let numToGenerate = targetCount - class1.length;
        
        if (numToGenerate > 0 && class1.length > 0) {
            let k = Math.min(3, class1.length);
            
            for (let i = 0; i < numToGenerate; i++) {
                // Select random seed minority sample
                let idx = Math.floor(rand() * class1.length);
                let sample = class1[idx];
                
                // Find nearest neighbors in Class 1
                let dists = [];
                for (let j = 0; j < class1.length; j++) {
                    if (j === idx && class1.length > 1) continue;
                    let other = class1[j];
                    let d = Math.hypot(
                        sample.features[1] - other.features[1],
                        sample.features[2] - other.features[2]
                    );
                    dists.push({ index: j, dist: d });
                }
                
                // Sort by distance
                dists.sort((a, b) => a.dist - b.dist);
                
                // Pick random neighbor from top K
                let neighborIdx = dists[Math.floor(rand() * Math.min(k, dists.length))].index;
                let neighbor = class1[neighborIdx];
                
                // Interpolate features
                let lambda = rand();
                let synAge = sample.age + lambda * (neighbor.age - sample.age);
                let synSbp = sample.sbp + lambda * (neighbor.sbp - sample.sbp);
                
                let synF1 = sample.features[1] + lambda * (neighbor.features[1] - sample.features[1]);
                let synF2 = sample.features[2] + lambda * (neighbor.features[2] - sample.features[2]);
                
                balancedClass1.push({
                    y: 1,
                    age: synAge,
                    sbp: synSbp,
                    features: [1.0, synF1, synF2],
                    isSynthetic: true
                });
            }
        }
        balancedDataset = [...class0, ...balancedClass1];
    }
}

// ==========================================
// GLM Solver (Binary Logistic Regression)
// ==========================================
function fitGLM() {
    let X = balancedDataset.map(d => d.features);
    let y = balancedDataset.map(d => d.y);
    let n = X.length;
    
    // Initialize coefficients [intercept, beta_age, beta_sbp]
    let beta = [0.0, 0.0, 0.0];
    let momentum = [0.0, 0.0, 0.0];
    
    let lr = 0.2;       // Learning rate
    let mu = 0.9;       // Momentum coefficient
    let epochs = 150;
    
    if (n === 0) {
        fittedBeta = beta;
        return;
    }
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        let grad = [0.0, 0.0, 0.0];
        
        for (let i = 0; i < n; i++) {
            let eta = beta[0] * X[i][0] + beta[1] * X[i][1] + beta[2] * X[i][2];
            eta = Math.max(-6.0, Math.min(6.0, eta)); // Clip linear predictor for numerical stability
            
            let p = 0.5;
            let dpdeta = 0.0;
            
            if (linkFunction === 'logit') {
                p = 1.0 / (1.0 + Math.exp(-eta));
                dpdeta = p * (1.0 - p);
            } 
            else if (linkFunction === 'probit') {
                p = normalCDF(eta);
                dpdeta = normalPDF(eta);
            } 
            else if (linkFunction === 'cloglog') {
                let e_eta = Math.exp(eta);
                p = 1.0 - Math.exp(-e_eta);
                dpdeta = e_eta * Math.exp(-e_eta);
            }
            
            // Clip probability to prevent divide by zero
            p = Math.max(1e-5, Math.min(1.0 - 1e-5, p));
            
            // GLM Gradient weight: (y - p) / (p * (1 - p)) * dp/deta
            let weight = (y[i] - p) / (p * (1.0 - p)) * dpdeta;
            
            grad[0] += weight * X[i][0];
            grad[1] += weight * X[i][1];
            grad[2] += weight * X[i][2];
        }
        
        // Update coefficients with momentum and ridge regularization (L2)
        for (let j = 0; j < 3; j++) {
            let g = grad[j] / n;
            g -= 0.01 * beta[j]; // Ridge penalty of 0.01
            
            momentum[j] = mu * momentum[j] + lr * g;
            beta[j] += momentum[j];
        }
    }
    
    fittedBeta = beta;
}

// Predict probability for a standardized feature vector
function predictProb(features) {
    let eta = fittedBeta[0] * features[0] + fittedBeta[1] * features[1] + fittedBeta[2] * features[2];
    eta = Math.max(-6.0, Math.min(6.0, eta));
    
    if (linkFunction === 'logit') {
        return 1.0 / (1.0 + Math.exp(-eta));
    } 
    else if (linkFunction === 'probit') {
        return normalCDF(eta);
    } 
    else { // cloglog
        return 1.0 - Math.exp(-Math.exp(eta));
    }
}

// ==========================================
// Metrics & ROC Computations
// ==========================================
function calculatePerformance() {
    let yTrue = balancedDataset.map(d => d.y);
    let yPred = balancedDataset.map(d => predictProb(d.features));
    let n = yTrue.length;
    
    if (n === 0) {
        return { accuracy: 0, sensitivity: 0, specificity: 0, auc: 0.5, tn: 0, fp: 0, fn: 0, tp: 0 };
    }
    
    // 1. Calculate Confusion Matrix at threshold 0.5
    let tn = 0, fp = 0, fn = 0, tp = 0;
    for (let i = 0; i < n; i++) {
        let actual = yTrue[i];
        let predicted = yPred[i] >= 0.5 ? 1 : 0;
        
        if (actual === 0 && predicted === 0) tn++;
        else if (actual === 0 && predicted === 1) fp++;
        else if (actual === 1 && predicted === 0) fn++;
        else if (actual === 1 && predicted === 1) tp++;
    }
    
    let totalActual0 = tn + fp;
    let totalActual1 = fn + tp;
    
    let accuracy = (tn + tp) / n;
    let sensitivity = totalActual1 > 0 ? tp / totalActual1 : 0.0;
    let specificity = totalActual0 > 0 ? tn / totalActual0 : 0.0;
    
    // 2. Compute ROC and AUC
    let samples = [];
    for (let i = 0; i < n; i++) {
        samples.push({ trueVal: yTrue[i], predVal: yPred[i] });
    }
    samples.sort((a, b) => b.predVal - a.predVal); // Sort descending
    
    let rocPoints = [{ x: 0, y: 0 }];
    let currentTp = 0;
    let currentFp = 0;
    let auc = 0.0;
    let lastFpr = 0.0;
    
    for (let i = 0; i < samples.length; i++) {
        if (samples[i].trueVal === 1) {
            currentTp++;
        } else {
            currentFp++;
            let tpr = totalActual1 > 0 ? currentTp / totalActual1 : 0.0;
            let fpr = totalActual0 > 0 ? currentFp / totalActual0 : 0.0;
            auc += (fpr - lastFpr) * tpr;
            lastFpr = fpr;
        }
        rocPoints.push({
            x: totalActual0 > 0 ? currentFp / totalActual0 : 0.0,
            y: totalActual1 > 0 ? currentTp / totalActual1 : 0.0
        });
    }
    rocPoints.push({ x: 1, y: 1 });
    
    // Handle bounds
    auc = Math.max(0.0, Math.min(1.0, auc));
    aucValue = auc;
    
    return {
        accuracy: accuracy,
        sensitivity: sensitivity,
        specificity: specificity,
        auc: auc,
        rocPoints: rocPoints,
        tn: tn,
        fp: fp,
        fn: fn,
        tp: tp
    };
}

// ==========================================
// UI Updates & Chart Rendering
// ==========================================
function updateUI() {
    // 1. Data counts
    let origClass0 = originalDataset.filter(d => d.y === 0).length;
    let origClass1 = originalDataset.filter(d => d.y === 1).length;
    
    let balClass0 = balancedDataset.filter(d => d.y === 0).length;
    let balClass1 = balancedDataset.filter(d => d.y === 1).length;
    let balTotal = balancedDataset.length;
    
    // 2. Format Imbalance Ratios
    let origRatioText = origClass1 > 0 ? `1 : ${(origClass0 / origClass1).toFixed(1)}` : '1 : N/A';
    let balRatioText = balClass1 > 0 ? `1 : ${(balClass0 / balClass1).toFixed(1)}` : '1 : N/A';
    
    // 3. Update Metric Cards
    document.getElementById('metric-total-samples').textContent = balTotal.toLocaleString();
    document.getElementById('sub-total-samples').textContent = `Original: n = ${sampleSize}`;
    
    document.getElementById('metric-class0-count').textContent = balClass0.toLocaleString();
    document.getElementById('sub-class0-pct').textContent = `${((balClass0 / balTotal) * 100).toFixed(1)}% of total`;
    
    document.getElementById('metric-class1-count').textContent = balClass1.toLocaleString();
    document.getElementById('sub-class1-pct').textContent = `${((balClass1 / balTotal) * 100).toFixed(1)}% of total`;
    
    document.getElementById('metric-imbalance-ratio').textContent = balRatioText;
    document.getElementById('sub-imbalance-orig').textContent = `Original: ${origRatioText}`;
    
    // 4. Calculate model predictions & metrics
    let perf = calculatePerformance();
    
    // 5. Update Confusion Matrix Cells (Heatmap rates)
    let totalActual0 = perf.tn + perf.fp;
    let totalActual1 = perf.fn + perf.tp;
    
    let tnr = totalActual0 > 0 ? perf.tn / totalActual0 : 0;
    let fpr = totalActual0 > 0 ? perf.fp / totalActual0 : 0;
    let fnr = totalActual1 > 0 ? perf.fn / totalActual1 : 0;
    let tpr = totalActual1 > 0 ? perf.tp / totalActual1 : 0;
    
    updateMatrixCell('cell-tn', 'val-tn', 'cnt-tn', tnr, perf.tn, '--opacity-tn');
    updateMatrixCell('cell-fp', 'val-fp', 'cnt-fp', fpr, perf.fp, '--opacity-fp');
    updateMatrixCell('cell-fn', 'val-fn', 'cnt-fn', fnr, perf.fn, '--opacity-fn');
    updateMatrixCell('cell-tp', 'val-tp', 'cnt-tp', tpr, perf.tp, '--opacity-tp');
    
    // 6. Update Information Footer Summary
    document.getElementById('footer-auc').textContent = perf.auc.toFixed(3);
    document.getElementById('footer-accuracy').textContent = `${(perf.accuracy * 100).toFixed(1)}%`;
    document.getElementById('footer-sensitivity').textContent = `${(perf.sensitivity * 100).toFixed(1)}%`;
    document.getElementById('footer-specificity').textContent = `${(perf.specificity * 100).toFixed(1)}%`;
    
    let linkLabel = linkFunction === 'logit' ? 'Logit' : (linkFunction === 'probit' ? 'Probit' : 'Cloglog');
    let balanceLabel = 'None';
    if (balancingMethod === 'oversampling') balanceLabel = 'Oversample';
    else if (balancingMethod === 'undersampling') balanceLabel = 'Undersample';
    else if (balancingMethod === 'smote') balanceLabel = 'SMOTE';
    
    document.getElementById('footer-link-badge').textContent = linkLabel;
    document.getElementById('footer-balance-badge').textContent = balanceLabel;
    
    // 7. Render/Update Charts
    renderClassDistChart(origClass0, origClass1, balClass0, balClass1);
    renderLinkFunctionChart();
    renderROCChart(perf.rocPoints);
}

function updateMatrixCell(cellId, valId, cntId, rate, count, opacityVar) {
    let cell = document.getElementById(cellId);
    let valEl = document.getElementById(valId);
    let cntEl = document.getElementById(cntId);
    
    valEl.textContent = `${(rate * 100).toFixed(1)}%`;
    cntEl.textContent = `n = ${count}`;
    
    // Set cell opacity variable dynamically
    // Scale rate to give a visible color transition (e.g. from 0.05 to 0.85 opacity)
    let opacity = 0.05 + 0.80 * rate;
    cell.style.setProperty(opacityVar, opacity);
    
    // Improve text contrast based on background color intensity
    if (opacity > 0.5) {
        cell.style.color = '#ffffff';
        // Overwrite standard primary/danger color for text inside dark background cells
        cell.style.setProperty('--text-tn', '#ffffff');
        cell.style.setProperty('--text-tp', '#ffffff');
        cell.style.setProperty('--text-fp', '#ffffff');
        cell.style.setProperty('--text-fn', '#ffffff');
    } else {
        cell.style.color = 'var(--text-primary)';
        cell.style.setProperty('--text-tn', 'var(--color-primary)');
        cell.style.setProperty('--text-tp', 'var(--color-primary)');
        cell.style.setProperty('--text-fp', 'var(--color-danger)');
        cell.style.setProperty('--text-fn', 'var(--color-danger)');
    }
}

// ==========================================
// Chart.js Implementations
// ==========================================

function renderClassDistChart(orig0, orig1, bal0, bal1) {
    let ctx = document.getElementById('classDistChart').getContext('2d');
    
    let chartData = {
        labels: ['Original (Imbalanced)', 'Balanced Model Fit'],
        datasets: [
            {
                label: 'Class 0 (No Risk)',
                data: [orig0, bal0],
                backgroundColor: 'rgba(37, 99, 235, 0.7)',
                borderColor: '#2563eb',
                borderWidth: 1.5,
                borderRadius: 4
            },
            {
                label: 'Class 1 (High Risk)',
                data: [orig1, bal1],
                backgroundColor: 'rgba(225, 29, 72, 0.7)',
                borderColor: '#e11d48',
                borderWidth: 1.5,
                borderRadius: 4
            }
        ]
    };
    
    if (classDistChart) {
        classDistChart.data = chartData;
        classDistChart.update();
    } else {
        classDistChart = new Chart(ctx, {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { boxWidth: 12, font: { family: 'Inter', weight: 600 } }
                    },
                    tooltip: {
                        callbacks: {
                            afterBody: function(items) {
                                if (balancingMethod === 'smote' && items[0].dataIndex === 1 && items[0].datasetIndex === 1) {
                                    return '\n*Includes SMOTE-synthesised samples';
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { 
                        beginAtZero: true,
                        title: { display: true, text: 'Sample Count', font: { weight: 600 } }
                    }
                }
            }
        });
    }
}

function renderLinkFunctionChart() {
    let ctx = document.getElementById('fittedCurvesChart').getContext('2d');
    
    // Generate eta points from -4.0 to 4.0
    let etaValues = [];
    let logitProbs = [];
    let probitProbs = [];
    let cloglogProbs = [];
    
    for (let eta = -4.0; eta <= 4.05; eta += 0.1) {
        etaValues.push(Number(eta.toFixed(1)));
        
        // Inverse link functions
        let pLog = 1.0 / (1.0 + Math.exp(-eta));
        let pProb = normalCDF(eta);
        let pClog = 1.0 - Math.exp(-Math.exp(eta));
        
        logitProbs.push(pLog);
        probitProbs.push(pProb);
        cloglogProbs.push(pClog);
    }
    
    // Determine widths based on active selection
    let logitWidth = linkFunction === 'logit' ? 4.5 : 1.5;
    let probitWidth = linkFunction === 'probit' ? 4.5 : 1.5;
    let cloglogWidth = linkFunction === 'cloglog' ? 4.5 : 1.5;
    
    let datasets = [
        {
            label: 'Logit (Logistic)',
            data: logitProbs,
            borderColor: '#2563eb', // Primary Blue
            borderWidth: logitWidth,
            pointRadius: 0,
            fill: false,
            // Solid Line
            borderDash: []
        },
        {
            label: 'Probit',
            data: probitProbs,
            borderColor: '#0d9488', // Emerald Teal
            borderWidth: probitWidth,
            pointRadius: 0,
            fill: false,
            // Dashed Line
            borderDash: [5, 4]
        },
        {
            label: 'Cloglog',
            data: cloglogProbs,
            borderColor: '#ea580c', // Orange/Amber
            borderWidth: cloglogWidth,
            pointRadius: 0,
            fill: false,
            // Dotted/Short dash Line
            borderDash: [2, 3]
        }
    ];
    
    if (fittedCurvesChart) {
        fittedCurvesChart.data.datasets = datasets;
        fittedCurvesChart.update();
    } else {
        fittedCurvesChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: etaValues,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { boxWidth: 18, font: { family: 'Inter', weight: 600 } }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Linear Predictor (η)', font: { weight: 600 } },
                        ticks: { maxTicksLimit: 9 }
                    },
                    y: {
                        min: 0,
                        max: 1.0,
                        title: { display: true, text: 'Probability P(Y = 1)', font: { weight: 600 } }
                    }
                }
            }
        });
    }
}

function renderROCChart(rocPoints) {
    let ctx = document.getElementById('rocChart').getContext('2d');
    
    // Create random classifier baseline diagonal
    let baseline = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
    
    let chartData = {
        datasets: [
            {
                label: `Model ROC (AUC: ${aucValue.toFixed(3)})`,
                data: rocPoints,
                borderColor: '#4f46e5', // Indigo
                borderWidth: 3,
                pointRadius: 0,
                showLine: true,
                fill: false,
                borderDash: []
            },
            {
                label: 'Random Classifier (AUC: 0.500)',
                data: baseline,
                borderColor: '#94a3b8', // Gray Muted
                borderWidth: 1.5,
                pointRadius: 0,
                showLine: true,
                fill: false,
                borderDash: [4, 4]
            }
        ]
    };
    
    if (rocChart) {
        rocChart.data = chartData;
        rocChart.update();
    } else {
        rocChart = new Chart(ctx, {
            type: 'scatter',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { boxWidth: 12, font: { family: 'Inter', weight: 600 } }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: 0,
                        max: 1.0,
                        title: { display: true, text: '1 - Specificity (FPR)', font: { weight: 600 } }
                    },
                    y: {
                        min: 0,
                        max: 1.0,
                        title: { display: true, text: 'Sensitivity (TPR)', font: { weight: 600 } }
                    }
                }
            }
        });
    }
}

// ==========================================
// Controller Event Bindings & Init
// ==========================================
function processPipeline() {
    generateOriginalData();
    balanceData();
    fitGLM();
    updateUI();
}

function setupEventListeners() {
    // 1. Sample Size Slider
    const sizeSlider = document.getElementById('sample-size-slider');
    const sizeValue = document.getElementById('sample-size-val');
    sizeSlider.addEventListener('input', function(e) {
        sampleSize = parseInt(e.target.value);
        sizeValue.textContent = `n = ${sampleSize}`;
        processPipeline();
    });
    
    // 2. Class Imbalance Slider
    const imbSlider = document.getElementById('imbalance-ratio-slider');
    const imbValue = document.getElementById('imbalance-ratio-val');
    imbSlider.addEventListener('input', function(e) {
        let pct = parseInt(e.target.value);
        imbalanceRatio = pct / 100.0;
        imbValue.textContent = `${pct}%`;
        processPipeline();
    });
    
    // 3. Balancing Method Selector Buttons
    const balanceSelector = document.getElementById('balancing-selector');
    const balanceBtns = balanceSelector.querySelectorAll('.toggle-btn');
    balanceBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            balanceBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            balancingMethod = this.getAttribute('data-value');
            processPipeline();
        });
    });
    
    // 4. Link Function Selector Buttons
    const linkSelector = document.getElementById('link-selector');
    const linkBtns = linkSelector.querySelectorAll('.toggle-btn');
    linkBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            linkBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            linkFunction = this.getAttribute('data-value');
            processPipeline();
        });
    });
}

// Start Dashboard
window.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    processPipeline();
});
