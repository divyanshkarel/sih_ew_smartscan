import 'dart:convert';
import 'package:http/http.dart' as http;

/// Electronic Warfare Simulation Metrics Model
class SimulationMetrics {
  final double pd;
  final double pfa;
  final int cumulativeHits;
  final int cumulativeMisses;
  final int totalScans;
  final double cumulativeReward;
  final double interceptTimeErrorUs;

  const SimulationMetrics({
    required this.pd,
    required this.pfa,
    required this.cumulativeHits,
    required this.cumulativeMisses,
    required this.totalScans,
    required this.cumulativeReward,
    required this.interceptTimeErrorUs,
  });

  factory SimulationMetrics.empty() {
    return const SimulationMetrics(
      pd: 0.0,
      pfa: 0.0,
      cumulativeHits: 0,
      cumulativeMisses: 0,
      totalScans: 0,
      cumulativeReward: 0.0,
      interceptTimeErrorUs: 0.0,
    );
  }

  factory SimulationMetrics.fromJson(Map<String, dynamic> json) {
    return SimulationMetrics(
      pd: (json['pd'] as num?)?.toDouble() ?? 0.0,
      pfa: (json['pfa'] as num?)?.toDouble() ?? 0.0,
      cumulativeHits: (json['cumulative_hits'] as num?)?.toInt() ?? 0,
      cumulativeMisses: (json['cumulative_misses'] as num?)?.toInt() ?? 0,
      totalScans: (json['total_scans'] as num?)?.toInt() ?? 0,
      cumulativeReward: (json['cumulative_reward'] as num?)?.toDouble() ?? 0.0,
      interceptTimeErrorUs:
          (json['intercept_time_error_us'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

/// Single Simulation Step Result Model
class StepResult {
  final int step;
  final String strategy;
  final int chosenBand;
  final bool isHit;
  final double reward;
  final double baseReward;
  final double hardwarePenalty;
  final int jumpDistance;
  final bool terminated;
  final double inferenceLatencyUs;
  final SimulationMetrics metrics;

  const StepResult({
    required this.step,
    required this.strategy,
    required this.chosenBand,
    required this.isHit,
    required this.reward,
    required this.baseReward,
    required this.hardwarePenalty,
    required this.jumpDistance,
    required this.terminated,
    required this.inferenceLatencyUs,
    required this.metrics,
  });

  factory StepResult.initial(String strategy) {
    return StepResult(
      step: 0,
      strategy: strategy,
      chosenBand: 0,
      isHit: false,
      reward: 0.0,
      baseReward: 0.0,
      hardwarePenalty: 0.0,
      jumpDistance: 0,
      terminated: false,
      inferenceLatencyUs: 0.0,
      metrics: SimulationMetrics.empty(),
    );
  }

  factory StepResult.fromJson(Map<String, dynamic> json) {
    return StepResult(
      step: (json['step'] as num?)?.toInt() ?? 0,
      strategy: (json['strategy'] as String?) ?? 'ml_agent',
      chosenBand: (json['chosen_band'] as num?)?.toInt() ?? 0,
      isHit: (json['is_hit'] as bool?) ?? false,
      reward: (json['reward'] as num?)?.toDouble() ?? 0.0,
      baseReward: (json['base_reward'] as num?)?.toDouble() ?? 0.0,
      hardwarePenalty: (json['hardware_penalty'] as num?)?.toDouble() ?? 0.0,
      jumpDistance: (json['jump_distance'] as num?)?.toInt() ?? 0,
      terminated: (json['terminated'] as bool?) ?? false,
      inferenceLatencyUs:
          (json['inference_latency_us'] as num?)?.toDouble() ?? 0.0,
      metrics: SimulationMetrics.fromJson(
        (json['metrics'] as Map<String, dynamic>?) ?? {},
      ),
    );
  }
}

/// Dual Step Result for Concurrent Live Animation
class DualStepResult {
  final int step;
  final StepResult linear;
  final StepResult mlAgent;

  const DualStepResult({
    required this.step,
    required this.linear,
    required this.mlAgent,
  });

  factory DualStepResult.fromJson(Map<String, dynamic> json) {
    return DualStepResult(
      step: (json['step'] as num?)?.toInt() ?? 0,
      linear: StepResult.fromJson((json['linear'] as Map<String, dynamic>?) ?? {}),
      mlAgent: StepResult.fromJson(
        (json['ml_agent'] as Map<String, dynamic>?) ?? {},
      ),
    );
  }
}

/// Simulation Reset Response Model
class ResetResponse {
  final String status;
  final int step;
  final int totalSteps;
  final int numBands;
  final int observationDim;
  final SimulationMetrics linearMetrics;
  final SimulationMetrics mlMetrics;

  const ResetResponse({
    required this.status,
    required this.step,
    required this.totalSteps,
    required this.numBands,
    required this.observationDim,
    required this.linearMetrics,
    required this.mlMetrics,
  });

  factory ResetResponse.fromJson(Map<String, dynamic> json) {
    return ResetResponse(
      status: (json['status'] as String?) ?? 'ok',
      step: (json['step'] as num?)?.toInt() ?? 0,
      totalSteps: (json['total_steps'] as num?)?.toInt() ?? 6823,
      numBands: (json['num_bands'] as num?)?.toInt() ?? 20,
      observationDim: (json['observation_dim'] as num?)?.toInt() ?? 46,
      linearMetrics: SimulationMetrics.fromJson(
        (json['linear_metrics'] as Map<String, dynamic>?) ?? {},
      ),
      mlMetrics: SimulationMetrics.fromJson(
        (json['ml_metrics'] as Map<String, dynamic>?) ?? {},
      ),
    );
  }
}

/// API Service connecting to FastAPI Backend on http://127.0.0.1:8005
class ApiService {
  final String baseUrl;
  final http.Client _client;

  ApiService({
    this.baseUrl = 'http://127.0.0.1:8005',
    http.Client? client,
  }) : _client = client ?? http.Client();

  /// Reset environment to step 0
  Future<ResetResponse> resetSimulation() async {
    final uri = Uri.parse('$baseUrl/reset');
    final response = await _client.get(
      uri,
      headers: {'Accept': 'application/json'},
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return ResetResponse.fromJson(data);
    } else {
      throw Exception(
        'Failed to reset simulation. Status: ${response.statusCode}, Body: ${response.body}',
      );
    }
  }

  /// Execute single step for a given strategy ('linear' or 'ml_agent')
  Future<StepResult> stepSimulation(String strategy) async {
    final uri = Uri.parse('$baseUrl/step');
    final response = await _client.post(
      uri,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: jsonEncode({'strategy': strategy}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return StepResult.fromJson(data);
    } else {
      throw Exception(
        'Failed to step simulation ($strategy). Status: ${response.statusCode}, Body: ${response.body}',
      );
    }
  }

  /// Simultaneously step both strategies in lockstep
  Future<DualStepResult> stepBoth() async {
    final uri = Uri.parse('$baseUrl/step_both');
    final response = await _client.post(
      uri,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return DualStepResult.fromJson(data);
    } else {
      // Fallback: execute both via POST /step concurrently
      final futures = await Future.wait([
        stepSimulation('linear'),
        stepSimulation('ml_agent'),
      ]);
      return DualStepResult(
        step: futures[1].step,
        linear: futures[0],
        mlAgent: futures[1],
      );
    }
  }

  void dispose() {
    _client.close();
  }
}
