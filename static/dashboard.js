// Trading Bot Dashboard JavaScript
const socket = io();
let priceChart, volumeChart, performanceChart, signalChart;
let chartData = {
    prices: {},
    volumes: {},
    performance: [],
    signals: []
};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupEventListeners();
    populateTradingPairs();
});

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to dashboard server');
    updateConnectionStatus(true);
    socket.emit('request_update');
});

socket.on('disconnect', () => {
    console.log('Disconnected from dashboard server');
    updateConnectionStatus(false);
});

socket.on('dashboard_update', (data) => {
    updateDashboard(data);
    updateLastUpdate();
});

socket.on('new_signal', (signal) => {
    addSignal(signal);
    updateSignalChart(signal);
});

socket.on('alert', (alertData) => {
    showAlert(alertData.title, alertData.message, alertData.type);
});

// Initialize Chart.js charts
function initializeCharts() {
    // Price Chart
    const priceCtx = document.getElementById('priceChart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute'
                    }
                },
                y: {
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    display: true
                },
                title: {
                    display: true,
                    text: 'Price Movements'
                }
            }
        }
    });

    // Volume Chart
    const volumeCtx = document.getElementById('volumeChart').getContext('2d');
    volumeChart = new Chart(volumeCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: '24h Volume',
                data: [],
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Performance Chart
    const perfCtx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(perfCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });

    // Signal Chart
    const signalCtx = document.getElementById('signalChart').getContext('2d');
    signalChart = new Chart(signalCtx, {
        type: 'doughnut',
        data: {
            labels: ['Bullish', 'Bearish', 'Neutral'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(34, 197, 94, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(156, 163, 175, 0.8)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'Signal Distribution'
                }
            }
        }
    });
}

// Setup event listeners
function setupEventListeners() {
    // Manual signal button
    document.getElementById('send-signal-btn').addEventListener('click', toggleManualSignalForm);

    // Manual signal form
    document.getElementById('signal-form').addEventListener('submit', handleManualSignal);

    // Backtest button
    document.getElementById('run-backtest-btn').addEventListener('click', runBacktest);

    // Refresh data button
    document.getElementById('refresh-data-btn').addEventListener('click', refreshData);

    // Alert modal close
    document.querySelector('.close').addEventListener('click', hideAlert);
}

// Update dashboard with new data
function updateDashboard(data) {
    // Update system status
    document.getElementById('status-text').textContent = data.system_status || 'Unknown';
    document.getElementById('active-signals').textContent = data.signals ? data.signals.length : 0;
    document.getElementById('success-rate').textContent = data.performance?.win_rate ? `${data.performance.win_rate.toFixed(1)}%` : '--%';
    document.getElementById('total-pnl').textContent = data.performance?.total_return ? `$${data.performance.total_return.toFixed(2)}` : '$0.00';

    // Update market data table and charts
    updateMarketData(data.market_data);

    // Update performance metrics and chart
    updatePerformanceData(data.performance);

    // Update signals
    updateSignalsList(data.signals);
}

// Update market data table and charts
function updateMarketData(marketData) {
    const marketBody = document.getElementById('market-data-body');
    marketBody.innerHTML = '';

    const pairs = Object.keys(marketData || {});
    const prices = [];
    const volumes = [];

    pairs.forEach(pair => {
        const info = marketData[pair];

        // Update table
        const row = document.createElement('tr');
        const signalClass = getSignalClass(info.signal);
        row.innerHTML = `
            <td>${pair}</td>
            <td>${info.price ? info.price.toFixed(2) : 'N/A'}</td>
            <td class="${info.change_24h >= 0 ? 'positive' : 'negative'}">
                ${info.change_24h ? info.change_24h.toFixed(2) : 0}%
            </td>
            <td>${info.volume_24h ? info.volume_24h.toFixed(2) : 'N/A'}</td>
            <td>${info.funding_rate ? info.funding_rate.toFixed(4) : 'N/A'}</td>
            <td><span class="signal-badge ${signalClass}">${info.signal || 'NEUTRAL'}</span></td>
        `;
        marketBody.appendChild(row);

        // Prepare chart data
        if (info.price) prices.push(info.price);
        if (info.volume_24h) volumes.push(info.volume_24h);
    });

    // Update volume chart
    volumeChart.data.labels = pairs;
    volumeChart.data.datasets[0].data = volumes;
    volumeChart.update();

    // Update price chart (simplified - in real implementation, you'd track historical prices)
    if (priceChart.data.datasets.length !== pairs.length) {
        priceChart.data.datasets = pairs.map((pair, index) => ({
            label: pair,
            data: [marketData[pair]?.price || 0],
            borderColor: `hsl(${index * 360 / pairs.length}, 70%, 50%)`,
            backgroundColor: `hsl(${index * 360 / pairs.length}, 70%, 50%, 0.1)`,
            tension: 0.1
        }));
    }
    priceChart.update();
}

// Update performance data and chart
function updatePerformanceData(performance) {
    const perfList = document.getElementById('performance-list');
    perfList.innerHTML = '';

    if (performance) {
        Object.entries(performance).forEach(([key, value]) => {
            const item = document.createElement('li');
            let displayValue = value;
            if (typeof value === 'number') {
                if (key.includes('rate') || key.includes('return')) {
                    displayValue = `${value.toFixed(2)}${key.includes('rate') ? '%' : ''}`;
                } else if (key.includes('value') || key.includes('balance')) {
                    displayValue = `$${value.toFixed(2)}`;
                } else {
                    displayValue = value.toFixed(2);
                }
            }
            item.innerHTML = `<strong>${key.replace('_', ' ').toUpperCase()}:</strong> ${displayValue}`;
            perfList.appendChild(item);
        });

        // Update performance chart (simplified)
        if (performance.final_portfolio_value) {
            performanceChart.data.labels.push(new Date().toLocaleTimeString());
            performanceChart.data.datasets[0].data.push(performance.final_portfolio_value);
            if (performanceChart.data.labels.length > 20) {
                performanceChart.data.labels.shift();
                performanceChart.data.datasets[0].data.shift();
            }
            performanceChart.update();
        }
    }
}

// Update signals list
function updateSignalsList(signals) {
    const signalsList = document.getElementById('signals-list');
    signalsList.innerHTML = '';

    if (signals && signals.length > 0) {
        signals.slice(-10).forEach(signal => {
            const item = document.createElement('li');
            const signalClass = getSignalClass(signal.type);
            item.innerHTML = `
                <span class="signal-time">${signal.timestamp || 'N/A'}</span>
                <span class="signal-pair">${signal.pair || 'Unknown'}</span>
                <span class="signal-badge ${signalClass}">${signal.type || 'NEUTRAL'}</span>
                <span class="signal-price">$${signal.price ? signal.price.toFixed(2) : 'N/A'}</span>
                <span class="signal-leverage">${signal.leverage || 1}x</span>
                <span class="signal-confidence">${signal.confidence || 'Medium'}</span>
            `;
            signalsList.appendChild(item);
        });
    }
}

// Add new signal to list
function addSignal(signal) {
    const signalsList = document.getElementById('signals-list');
    const item = document.createElement('li');
    const signalClass = getSignalClass(signal.type);
    item.innerHTML = `
        <span class="signal-time">${signal.timestamp || new Date().toISOString()}</span>
        <span class="signal-pair">${signal.pair || 'Unknown'}</span>
        <span class="signal-badge ${signalClass}">${signal.type || 'NEUTRAL'}</span>
        <span class="signal-price">$${signal.price ? signal.price.toFixed(2) : 'N/A'}</span>
        <span class="signal-leverage">${signal.leverage || 1}x</span>
        <span class="signal-confidence">${signal.confidence || 'Medium'}</span>
    `;
    signalsList.insertBefore(item, signalsList.firstChild);

    // Keep only last 20 signals
    while (signalsList.children.length > 20) {
        signalsList.removeChild(signalsList.lastChild);
    }

    // Update signal chart
    updateSignalChart(signal);
}

// Update signal distribution chart
function updateSignalChart(signal) {
    // This is a simplified implementation
    // In a real system, you'd track signal counts over time
    const signals = document.querySelectorAll('#signals-list .signal-badge');
    let bullish = 0, bearish = 0, neutral = 0;

    signals.forEach(badge => {
        const text = badge.textContent.toLowerCase();
        if (text.includes('bull')) bullish++;
        else if (text.includes('bear')) bearish++;
        else neutral++;
    });

    signalChart.data.datasets[0].data = [bullish, bearish, neutral];
    signalChart.update();
}

// Utility functions
function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (connected) {
        statusElement.className = 'status-connected';
        statusElement.textContent = '● Connected';
    } else {
        statusElement.className = 'status-disconnected';
        statusElement.textContent = '● Disconnected';
    }
}

function updateLastUpdate() {
    const now = new Date();
    document.getElementById('last-update').textContent =
        `Last update: ${now.toLocaleTimeString()}`;
}

function getSignalClass(signal) {
    if (!signal) return 'neutral';
    const sig = signal.toLowerCase();
    if (sig.includes('bull')) return 'bullish';
    if (sig.includes('bear')) return 'bearish';
    return 'neutral';
}

function toggleManualSignalForm() {
    const form = document.getElementById('manual-signal-form');
    form.classList.toggle('hidden');
}

function handleManualSignal(event) {
    event.preventDefault();

    const pair = document.getElementById('manual-pair').value;
    const signalType = document.getElementById('manual-signal-type').value;

    if (!pair || !signalType) {
        showAlert('Error', 'Please select both pair and signal type', 'error');
        return;
    }

    // Send manual signal request
    fetch('/api/send-signal', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            pair: pair,
            signal_type: signalType,
            manual: true
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Success', data.message, 'success');
            document.getElementById('manual-signal-form').classList.add('hidden');
        } else {
            showAlert('Error', data.message, 'error');
        }
    })
    .catch(error => {
        showAlert('Error', 'Failed to send manual signal', 'error');
        console.error('Error:', error);
    });
}

function runBacktest() {
    const startDate = prompt('Enter start date (YYYY-MM-DD):', '2024-01-01');
    const endDate = prompt('Enter end date (YYYY-MM-DD):', new Date().toISOString().split('T')[0]);

    if (!startDate || !endDate) return;

    showAlert('Info', 'Running backtest...', 'info');

    fetch('/api/backtest', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            start_date: startDate,
            end_date: endDate,
            pairs: [] // Will use default pairs
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Success', 'Backtest completed successfully', 'success');
            // Refresh dashboard to show new performance data
            socket.emit('request_update');
        } else {
            showAlert('Error', data.message, 'error');
        }
    })
    .catch(error => {
        showAlert('Error', 'Failed to run backtest', 'error');
        console.error('Error:', error);
    });
}

function refreshData() {
    socket.emit('request_update');
    showAlert('Info', 'Refreshing data...', 'info');
}

function populateTradingPairs() {
    // This would typically fetch from the server
    const pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'XRPUSDT', 'LINKUSDT', 'DOGEUSDT'];
    const select = document.getElementById('manual-pair');

    pairs.forEach(pair => {
        const option = document.createElement('option');
        option.value = pair;
        option.textContent = pair;
        select.appendChild(option);
    });
}

function showAlert(title, message, type = 'info') {
    const modal = document.getElementById('alert-modal');
    const titleElement = document.getElementById('alert-title');
    const messageElement = document.getElementById('alert-message');

    titleElement.textContent = title;
    messageElement.textContent = message;

    modal.className = `modal ${type}`;
    modal.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideAlert();
    }, 5000);
}

function hideAlert() {
    document.getElementById('alert-modal').classList.add('hidden');
}

// Anomaly Detection Integration
let anomalyList, anomalySection;

// Initialize anomaly detection section
function initializeAnomalyDetection() {
    // Create anomaly section
    anomalySection = document.createElement('section');
    anomalySection.id = 'anomalies';
    anomalySection.innerHTML = '<h2>Anomaly Detection Alerts</h2>';

    // Create anomaly list
    anomalyList = document.createElement('ul');
    anomalyList.id = 'anomaly-list';
    anomalyList.innerHTML = '<li>Loading anomalies...</li>';
    anomalySection.appendChild(anomalyList);

    // Create anomaly controls
    const anomalyControls = document.createElement('div');
    anomalyControls.className = 'anomaly-controls';
    anomalyControls.innerHTML = `
        <button id="scan-anomalies-btn" class="btn btn-warning">Scan Anomalies Now</button>
        <button id="clear-anomalies-btn" class="btn btn-secondary">Clear Anomalies</button>
        <div class="anomaly-summary" id="anomaly-summary"></div>
    `;
    anomalySection.appendChild(anomalyControls);

    // Insert anomaly section before controls section
    const controlsSection = document.getElementById('controls');
    controlsSection.parentNode.insertBefore(anomalySection, controlsSection);

    // Setup anomaly event listeners
    document.getElementById('scan-anomalies-btn').addEventListener('click', scanAnomalies);
    document.getElementById('clear-anomalies-btn').addEventListener('click', clearAnomalies);
}

// Update anomaly list
function updateAnomalies(anomalies, summary) {
    anomalyList.innerHTML = '';

    if (!anomalies || anomalies.length === 0) {
        const li = document.createElement('li');
        li.textContent = 'No anomalies detected.';
        li.className = 'no-anomalies';
        anomalyList.appendChild(li);
    } else {
        anomalies.forEach(anomaly => {
            const li = document.createElement('li');
            li.className = `anomaly-item ${anomaly.severity.toLowerCase()}`;
            li.innerHTML = `
                <div class="anomaly-header">
                    <span class="anomaly-type">${anomaly.type}</span>
                    <span class="anomaly-severity ${anomaly.severity.toLowerCase()}">${anomaly.severity}</span>
                    <span class="anomaly-time">${new Date(anomaly.timestamp).toLocaleString()}</span>
                </div>
                <div class="anomaly-details">
                    <span class="anomaly-pair">${anomaly.trading_pair}</span>
                    <span class="anomaly-confidence">Confidence: ${(anomaly.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="anomaly-description">${anomaly.description}</div>
            `;
            anomalyList.appendChild(li);
        });
    }

    // Update summary
    updateAnomalySummary(summary);
}

// Update anomaly summary
function updateAnomalySummary(summary) {
    const summaryElement = document.getElementById('anomaly-summary');
    if (summary && Object.keys(summary).length > 0) {
        summaryElement.innerHTML = `
            <div class="summary-stats">
                <span class="total-anomalies">Total: ${summary.total_anomalies || 0}</span>
                <span class="high-severity">High: ${summary.by_severity?.HIGH || 0}</span>
                <span class="critical-severity">Critical: ${summary.by_severity?.CRITICAL || 0}</span>
            </div>
        `;
    } else {
        summaryElement.innerHTML = '<div class="summary-stats">No anomaly data available</div>';
    }
}

// Scan for anomalies manually
function scanAnomalies() {
    const button = document.getElementById('scan-anomalies-btn');
    const originalText = button.textContent;
    button.textContent = 'Scanning...';
    button.disabled = true;

    fetch('/api/scan-anomalies', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showAlert('Success', `Scan complete: Found ${data.anomalies.length} anomalies.`, 'success');
                // Refresh dashboard data
                socket.emit('request_update');
            } else {
                showAlert('Error', 'Scan failed: ' + data.message, 'error');
            }
        })
        .catch(error => {
            showAlert('Error', 'Error scanning anomalies: ' + error.message, 'error');
        })
        .finally(() => {
            button.textContent = originalText;
            button.disabled = false;
        });
}

// Clear anomalies
function clearAnomalies() {
    if (confirm('Are you sure you want to clear all anomaly data?')) {
        // This would typically call a server endpoint to clear anomalies
        showAlert('Info', 'Anomaly clearing not implemented in this demo', 'info');
    }
}

// Enhanced dashboard update to include anomalies
const originalUpdateDashboard = updateDashboard;
updateDashboard = function(data) {
    // Call original update function
    originalUpdateDashboard(data);

    // Update anomalies
    if (data.anomalies) {
        updateAnomalies(data.anomalies, data.anomaly_summary);
    }
};

// Initialize anomaly detection when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeAnomalyDetection();
});

// Initialize on load
initializeCharts();
setupEventListeners();
populateTradingPairs();
