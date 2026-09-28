import 'dart:async';
import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../widgets/metrics_ticker.dart';
import '../widgets/spectrum_visualizer.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final ApiService _apiService = ApiService();

  Timer? _simulationTimer;
  bool _isRunning = false;
  bool _isConnecting = false;
  bool _isConnected = false;
  String? _errorMessage;

  int _currentStep = 0;
  int _totalSteps = 6823;
  int _numBands = 20;

  StepResult _linearResult = StepResult.initial('linear');
  StepResult _mlResult = StepResult.initial('ml_agent');

  SimulationMetrics _linearMetrics = SimulationMetrics.empty();
  SimulationMetrics _mlMetrics = SimulationMetrics.empty();

  // Sliding history trails for both agents
  final List<bool> _linearHistory = [];
  final List<bool> _mlHistory = [];

  @override
  void initState() {
    super.initState();
    _initializeConnection();
  }

  @override
  void dispose() {
    _simulationTimer?.cancel();
    _apiService.dispose();
    super.dispose();
  }

  Future<void> _initializeConnection() async {
    setState(() {
      _isConnecting = true;
      _errorMessage = null;
    });

    try {
      final resetResp = await _apiService.resetSimulation();
      if (!mounted) return;
      setState(() {
        _isConnected = true;
        _isConnecting = false;
        _currentStep = resetResp.step;
        _totalSteps = resetResp.totalSteps;
        _numBands = resetResp.numBands;
        _linearMetrics = resetResp.linearMetrics;
        _mlMetrics = resetResp.mlMetrics;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isConnected = false;
        _isConnecting = false;
        _errorMessage =
            'Cannot connect to FastAPI backend at http://127.0.0.1:8005.\nEnsure "python api_backend.py" is running.';
      });
    }
  }

  void _toggleSimulation() {
    if (_isRunning) {
      _pauseSimulation();
    } else {
      _startSimulation();
    }
  }

  void _startSimulation() {
    if (!_isConnected) {
      _initializeConnection();
      return;
    }

    _simulationTimer?.cancel();
    setState(() {
      _isRunning = true;
    });

    // 10 Hz animation timer (100 ms interval)
    _simulationTimer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      _executeStep();
    });
  }

  void _pauseSimulation() {
    _simulationTimer?.cancel();
    if (!mounted) return;
    setState(() {
      _isRunning = false;
    });
  }

  Future<void> _resetSimulation() async {
    _pauseSimulation();
    try {
      final resetResp = await _apiService.resetSimulation();
      if (!mounted) return;
      setState(() {
        _currentStep = resetResp.step;
        _linearResult = StepResult.initial('linear');
        _mlResult = StepResult.initial('ml_agent');
        _linearMetrics = resetResp.linearMetrics;
        _mlMetrics = resetResp.mlMetrics;
        _linearHistory.clear();
        _mlHistory.clear();
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Reset failed: $e'),
          backgroundColor: Colors.red[800],
        ),
      );
    }
  }

  Future<void> _executeStep() async {
    try {
      final dualResult = await _apiService.stepBoth();
      if (!mounted) return;

      setState(() {
        _currentStep = dualResult.step;
        _linearResult = dualResult.linear;
        _mlResult = dualResult.mlAgent;
        _linearMetrics = dualResult.linear.metrics;
        _mlMetrics = dualResult.mlAgent.metrics;

        // Maintain 24-item visual activity history
        _linearHistory.insert(0, dualResult.linear.isHit);
        if (_linearHistory.length > 24) _linearHistory.removeLast();

        _mlHistory.insert(0, dualResult.mlAgent.isHit);
        if (_mlHistory.length > 24) _mlHistory.removeLast();
      });

      if (dualResult.mlAgent.terminated || dualResult.linear.terminated) {
        _pauseSimulation();
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Radar Episode Simulation Complete (All 6,823 Time Steps Evaluated)',
            ),
            backgroundColor: Color(0xFF00E676),
          ),
        );
      }
    } catch (e) {
      _pauseSimulation();
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Simulation step interrupted: $e';
      });
    }
  }

  Widget _buildHistoryTrail(List<bool> history) {
    if (history.isEmpty) {
      return Text(
        'AWAITING RADAR SWEEP TRAIL...',
        style: TextStyle(
          color: Colors.grey[600],
          fontSize: 10,
          fontFamily: 'monospace',
        ),
      );
    }

    return Row(
      children: [
        Text(
          'RECENT INTERCEPTS: ',
          style: TextStyle(
            color: Colors.grey[500],
            fontSize: 9,
            fontWeight: FontWeight.bold,
            fontFamily: 'monospace',
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: SizedBox(
            height: 14,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: history.length,
              separatorBuilder: (context, index) => const SizedBox(width: 3),
              itemBuilder: (context, idx) {
                final hit = history[idx];
                return Container(
                  width: 14,
                  height: 14,
                  decoration: BoxDecoration(
                    color: hit
                        ? const Color(0xFF00E676).withValues(alpha: 0.9)
                        : const Color(0xFFFF1744).withValues(alpha: 0.4),
                    borderRadius: BorderRadius.circular(2),
                  ),
                  child: Center(
                    child: Text(
                      hit ? 'H' : 'M',
                      style: const TextStyle(
                        color: Colors.black,
                        fontSize: 8,
                        fontWeight: FontWeight.bold,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildAgentColumn({
    required String title,
    required String subtitle,
    required Color accentColor,
    required StepResult result,
    required SimulationMetrics metrics,
    required List<bool> history,
    required bool isMl,
  }) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFF0E1526),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: accentColor.withValues(alpha: 0.3),
            width: 1.2,
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Title Header Banner
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title.toUpperCase(),
                      style: TextStyle(
                        color: accentColor,
                        fontSize: 14,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 1.2,
                        fontFamily: 'monospace',
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      style: TextStyle(
                        color: Colors.grey[500],
                        fontSize: 10,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: accentColor.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(
                      color: accentColor.withValues(alpha: 0.5),
                    ),
                  ),
                  child: Text(
                    isMl ? 'AUTONOMOUS RL' : 'OPEN-LOOP',
                    style: TextStyle(
                      color: accentColor,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.0,
                      fontFamily: 'monospace',
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 14),

            // Real-Time Spectrum Visualizer (20 Bands)
            SpectrumVisualizer(
              numBands: _numBands,
              activeBand: result.chosenBand,
              isHit: result.isHit,
              isScanning: _isRunning || result.step > 0,
              title: isMl ? 'RL Predicted Band' : 'Raster Scanned Band',
              accentColor: accentColor,
            ),

            const SizedBox(height: 14),

            // Performance Strip
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: const Color(0xFF080C16),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: const Color(0xFF1E293B)),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildStatItem('HITS', '${metrics.cumulativeHits}',
                      const Color(0xFF00E676)),
                  _buildStatItem('MISSES', '${metrics.cumulativeMisses}',
                      const Color(0xFFFF1744)),
                  _buildStatItem(
                    'PD',
                    '${(metrics.pd * 100).toStringAsFixed(1)}%',
                    const Color(0xFF00E5FF),
                  ),
                  _buildStatItem(
                    'PFA',
                    '${(metrics.pfa * 100).toStringAsFixed(1)}%',
                    const Color(0xFFFFB74D),
                  ),
                  _buildStatItem(
                    'REWARD',
                    metrics.cumulativeReward.toStringAsFixed(0),
                    Colors.white,
                  ),
                ],
              ),
            ),

            const SizedBox(height: 12),

            // Recent Intercept History Trail
            _buildHistoryTrail(history),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.grey[500],
            fontSize: 9,
            fontWeight: FontWeight.bold,
            fontFamily: 'monospace',
          ),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: 14,
            fontWeight: FontWeight.bold,
            fontFamily: 'monospace',
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final progress = _totalSteps > 0 ? (_currentStep / _totalSteps) : 0.0;

    return Scaffold(
      backgroundColor: const Color(0xFF060911),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0A0F1D),
        elevation: 2,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: const Color(0xFF00E5FF).withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(6),
              ),
              child: const Icon(
                Icons.radar,
                color: Color(0xFF00E5FF),
                size: 20,
              ),
            ),
            const SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'ELECTRONIC WARFARE COMMAND // SMART SCAN WAR ROOM',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w900,
                    letterSpacing: 1.2,
                    fontFamily: 'monospace',
                  ),
                ),
                Text(
                  'SIH-2026 BENCHMARK  •  TURING RADAR DATASET  •  ONNX EDGE ACCELERATION',
                  style: TextStyle(
                    color: Colors.grey[500],
                    fontSize: 9,
                    letterSpacing: 0.8,
                    fontFamily: 'monospace',
                  ),
                ),
              ],
            ),
          ],
        ),
        actions: [
          // Connection Status LED
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Row(
              children: [
                Container(
                  width: 10,
                  height: 10,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _isConnected
                        ? const Color(0xFF00E676)
                        : const Color(0xFFFF1744),
                    boxShadow: [
                      BoxShadow(
                        color: (_isConnected
                                ? const Color(0xFF00E676)
                                : const Color(0xFFFF1744))
                            .withValues(alpha: 0.8),
                        blurRadius: 8,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  _isConnected ? 'API CONNECTED (127.0.0.1:8005)' : 'API OFFLINE',
                  style: TextStyle(
                    color: _isConnected
                        ? const Color(0xFF00E676)
                        : const Color(0xFFFF1744),
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                    fontFamily: 'monospace',
                  ),
                ),
                const SizedBox(width: 12),
                IconButton(
                  icon: const Icon(Icons.refresh, size: 18),
                  tooltip: 'Reconnect API',
                  color: Colors.grey[400],
                  onPressed: _initializeConnection,
                ),
              ],
            ),
          ),
        ],
      ),
      body: _isConnecting
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(
                    color: Color(0xFF00E5FF),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'INITIALIZING ELECTRONIC WARFARE SIMULATOR...',
                    style: TextStyle(
                      color: Color(0xFF00E5FF),
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'monospace',
                    ),
                  ),
                ],
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Error Banner if offline
                  if (_errorMessage != null)
                    Container(
                      margin: const EdgeInsets.only(bottom: 14),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 10,
                      ),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFF1744).withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: const Color(0xFFFF1744)),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.warning,
                              color: Color(0xFFFF1744), size: 20),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              _errorMessage!,
                              style: const TextStyle(
                                color: Color(0xFFFF8A80),
                                fontSize: 11,
                                fontFamily: 'monospace',
                              ),
                            ),
                          ),
                          ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFFFF1744),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 6,
                              ),
                            ),
                            onPressed: _initializeConnection,
                            child: const Text('RETRY'),
                          ),
                        ],
                      ),
                    ),

                  // Simulation Progress Bar
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0E1526),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: const Color(0xFF1E293B)),
                    ),
                    child: Row(
                      children: [
                        Text(
                          'TIMELINE PROGRESS: ',
                          style: TextStyle(
                            color: Colors.grey[500],
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'monospace',
                          ),
                        ),
                        Expanded(
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: progress,
                              backgroundColor: const Color(0xFF161F30),
                              valueColor: const AlwaysStoppedAnimation<Color>(
                                Color(0xFF00E5FF),
                              ),
                              minHeight: 8,
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Text(
                          'STEP $_currentStep / $_totalSteps (${(progress * 100).toStringAsFixed(1)}%)',
                          style: const TextStyle(
                            color: Color(0xFF00E5FF),
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 14),

                  // Live Metrics Ticker Bar
                  MetricsTicker(
                    linearMetrics: _linearMetrics,
                    mlMetrics: _mlMetrics,
                    mlInferenceLatencyUs: _mlResult.inferenceLatencyUs > 0
                        ? _mlResult.inferenceLatencyUs
                        : 11.55,
                  ),

                  const SizedBox(height: 16),

                  // Side-by-Side Dual Column War Room Layout
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Left Column: Open-Loop Baseline
                      _buildAgentColumn(
                        title: 'Open-Loop Baseline',
                        subtitle: 'Legacy Sequential Raster Frequency Sweep',
                        accentColor: const Color(0xFFFFB74D), // Amber
                        result: _linearResult,
                        metrics: _linearMetrics,
                        history: _linearHistory,
                        isMl: false,
                      ),

                      const SizedBox(width: 16),

                      // Right Column: RL Smart Scan
                      _buildAgentColumn(
                        title: 'RL Smart Scan',
                        subtitle: 'Deep Q-Network + Agile Emitter Predictor',
                        accentColor: const Color(0xFF00E5FF), // Cyan
                        result: _mlResult,
                        metrics: _mlMetrics,
                        history: _mlHistory,
                        isMl: true,
                      ),
                    ],
                  ),
                ],
              ),
            ),
      // Floating Control Bar
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0xFF0F172A).withValues(alpha: 0.95),
          borderRadius: BorderRadius.circular(30),
          border: Border.all(
              color: const Color(0xFF00E5FF).withValues(alpha: 0.5)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.5),
              blurRadius: 16,
              spreadRadius: 2,
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Start / Pause FAB
            FloatingActionButton.extended(
              heroTag: 'play_pause_btn',
              onPressed: _isConnected ? _toggleSimulation : null,
              backgroundColor: _isRunning
                  ? const Color(0xFFFF9100) // Orange Pause
                  : const Color(0xFF00E676), // Green Start
              foregroundColor: Colors.black,
              icon: Icon(_isRunning ? Icons.pause : Icons.play_arrow),
              label: Text(
                _isRunning
                    ? 'PAUSE SIMULATION'
                    : 'START SIMULATION (10 Hz)',
                style: const TextStyle(
                  fontWeight: FontWeight.w900,
                  letterSpacing: 0.8,
                  fontFamily: 'monospace',
                ),
              ),
            ),

            const SizedBox(width: 12),

            // Step Forward (Single Step)
            IconButton(
              icon: const Icon(Icons.skip_next),
              tooltip: 'Single Step Forward',
              color: const Color(0xFF00E5FF),
              onPressed: _isConnected && !_isRunning ? _executeStep : null,
            ),

            // Reset Button
            IconButton(
              icon: const Icon(Icons.restart_alt),
              tooltip: 'Reset Episode to Step 0',
              color: Colors.grey[400],
              onPressed: _isConnected ? _resetSimulation : null,
            ),
          ],
        ),
      ),
    );
  }
}
