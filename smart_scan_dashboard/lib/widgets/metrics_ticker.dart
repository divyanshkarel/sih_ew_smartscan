import 'package:flutter/material.dart';
import '../services/api_service.dart';

/// MetricsTicker: High-level tactical KPI bar comparing live performance
/// between Open-Loop Baseline and RL Smart Scan.
class MetricsTicker extends StatelessWidget {
  final SimulationMetrics linearMetrics;
  final SimulationMetrics mlMetrics;
  final double mlInferenceLatencyUs;

  const MetricsTicker({
    super.key,
    required this.linearMetrics,
    required this.mlMetrics,
    this.mlInferenceLatencyUs = 11.55,
  });

  Widget _buildComparisonCard({
    required BuildContext context,
    required String label,
    required String linearValue,
    required String mlValue,
    required String deltaBadge,
    required Color deltaColor,
    required IconData icon,
    String? subtext,
  }) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: const Color(0xFF0F172A),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: const Color(0xFF1E293B)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.3),
              blurRadius: 6,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Title & Icon
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Icon(icon, size: 14, color: Colors.grey[400]),
                    const SizedBox(width: 6),
                    Text(
                      label.toUpperCase(),
                      style: TextStyle(
                        color: Colors.grey[400],
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.0,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: deltaColor.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(4),
                    border: Border.all(
                        color: deltaColor.withValues(alpha: 0.5), width: 0.8),
                  ),
                  child: Text(
                    deltaBadge,
                    style: TextStyle(
                      color: deltaColor,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'monospace',
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),

            // Side by Side Values
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                // Baseline side
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'LEGACY BASELINE',
                      style: TextStyle(
                        color: Colors.grey[500],
                        fontSize: 8,
                        fontFamily: 'monospace',
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      linearValue,
                      style: const TextStyle(
                        color: Color(0xFFFFB74D), // Amber / Legacy
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ),

                const Text(
                  'vs',
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 11,
                    fontStyle: FontStyle.italic,
                  ),
                ),

                // RL Agent side
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    const Text(
                      'RL SMART SCAN',
                      style: TextStyle(
                        color: Color(0xFF00E5FF),
                        fontSize: 8,
                        fontWeight: FontWeight.bold,
                        fontFamily: 'monospace',
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      mlValue,
                      style: const TextStyle(
                        color: Color(0xFF00E676), // Neon Green
                        fontSize: 18,
                        fontWeight: FontWeight.w900,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ),
              ],
            ),

            if (subtext != null) ...[
              const SizedBox(height: 6),
              Text(
                subtext,
                style: TextStyle(
                  color: Colors.grey[600],
                  fontSize: 9,
                  fontFamily: 'monospace',
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final pdDelta = (mlMetrics.pd - linearMetrics.pd) * 100;
    final pfaDelta = (mlMetrics.pfa - linearMetrics.pfa) * 100;
    final rewardMultiplier = linearMetrics.cumulativeReward > 0
        ? (mlMetrics.cumulativeReward / linearMetrics.cumulativeReward)
        : 1.0;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF0A0F1D),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF1E293B)),
      ),
      child: Row(
        children: [
          // 1. Probability of Detection (Pd)
          _buildComparisonCard(
            context: context,
            label: 'Probability of Detection (Pd)',
            linearValue: '${(linearMetrics.pd * 100).toStringAsFixed(1)}%',
            mlValue: '${(mlMetrics.pd * 100).toStringAsFixed(1)}%',
            deltaBadge: '${pdDelta >= 0 ? '+' : ''}${pdDelta.toStringAsFixed(1)}%',
            deltaColor: pdDelta >= 0
                ? const Color(0xFF00E676)
                : const Color(0xFFFF1744),
            icon: Icons.radar,
            subtext:
                'Hits: ${linearMetrics.cumulativeHits} vs ${mlMetrics.cumulativeHits}',
          ),

          const SizedBox(width: 10),

          // 2. Probability of False Alarm (Pfa)
          _buildComparisonCard(
            context: context,
            label: 'False Alarm Rate (Pfa)',
            linearValue: '${(linearMetrics.pfa * 100).toStringAsFixed(1)}%',
            mlValue: '${(mlMetrics.pfa * 100).toStringAsFixed(1)}%',
            deltaBadge: '${pfaDelta.toStringAsFixed(1)}%',
            deltaColor: pfaDelta <= 0
                ? const Color(0xFF00E676)
                : const Color(0xFFFF1744),
            icon: Icons.notifications_active,
            subtext:
                'Misses: ${linearMetrics.cumulativeMisses} vs ${mlMetrics.cumulativeMisses}',
          ),

          const SizedBox(width: 10),

          // 3. Mission Cumulative Reward
          _buildComparisonCard(
            context: context,
            label: 'Mission Reward Score',
            linearValue: linearMetrics.cumulativeReward.toStringAsFixed(0),
            mlValue: mlMetrics.cumulativeReward.toStringAsFixed(0),
            deltaBadge: '${rewardMultiplier.toStringAsFixed(2)}x',
            deltaColor: const Color(0xFF00E5FF),
            icon: Icons.military_tech,
            subtext: 'Includes frequency switching slew penalties',
          ),

          const SizedBox(width: 10),

          // 4. Edge Inference Latency
          _buildComparisonCard(
            context: context,
            label: 'Edge Compute Latency',
            linearValue: '0.0 us',
            mlValue: '${mlInferenceLatencyUs.toStringAsFixed(1)} us',
            deltaBadge: 'ONNX EDGE',
            deltaColor: const Color(0xFFE040FB),
            icon: Icons.speed,
            subtext: 'Single-thread microsecond inference',
          ),
        ],
      ),
    );
  }
}
