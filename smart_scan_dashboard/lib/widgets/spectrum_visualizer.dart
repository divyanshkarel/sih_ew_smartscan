import 'package:flutter/material.dart';

/// SpectrumVisualizer: Displays 20 vertical frequency spectrum bars
/// representing the discretized Electronic Warfare radar bands.
/// Highlights currently scanned band in blue, flashing Green for Hit and Red for Miss.
class SpectrumVisualizer extends StatelessWidget {
  final int numBands;
  final int activeBand;
  final bool isHit;
  final bool isScanning;
  final String title;
  final Color accentColor;
  final double minFreqMhz;
  final double maxFreqMhz;

  const SpectrumVisualizer({
    super.key,
    this.numBands = 20,
    required this.activeBand,
    required this.isHit,
    this.isScanning = false,
    required this.title,
    this.accentColor = const Color(0xFF00E5FF),
    this.minFreqMhz = 1174.7,
    this.maxFreqMhz = 11986.9,
  });

  Color _getBandColor(int bandIndex) {
    if (bandIndex != activeBand) {
      // Inactive frequency channel slot
      return const Color(0xFF161F30);
    }

    if (!isScanning) {
      return const Color(0xFF00E5FF); // Idle blue
    }

    // Active band scanned: Flash Green for Hit, Red for Miss
    if (isHit) {
      return const Color(0xFF00E676); // High-contrast neon green
    } else {
      return const Color(0xFFFF1744); // Crimson warning red
    }
  }

  List<BoxShadow> _getBandShadow(int bandIndex) {
    if (bandIndex != activeBand || !isScanning) {
      return [];
    }

    if (isHit) {
      return [
        BoxShadow(
          color: const Color(0xFF00E676).withValues(alpha: 0.8),
          blurRadius: 16,
          spreadRadius: 3,
        ),
      ];
    } else {
      return [
        BoxShadow(
          color: const Color(0xFFFF1744).withValues(alpha: 0.7),
          blurRadius: 14,
          spreadRadius: 2,
        ),
      ];
    }
  }

  double _getBandHeightFraction(int bandIndex) {
    if (bandIndex == activeBand) {
      return isHit ? 0.95 : 0.75;
    }
    // Ambient channel baseline height
    return 0.35 + ((bandIndex * 7) % 5) * 0.05;
  }

  double _getFrequencyForBand(int band) {
    final span = maxFreqMhz - minFreqMhz;
    return minFreqMhz + (band + 0.5) * (span / numBands);
  }

  @override
  Widget build(BuildContext context) {
    final activeFreq = _getFrequencyForBand(activeBand);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF0D1321),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isScanning && isHit
              ? const Color(0xFF00E676).withValues(alpha: 0.5)
              : const Color(0xFF1E293B),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.4),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header Status Badge
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: accentColor,
                      boxShadow: [
                        BoxShadow(
                          color: accentColor.withValues(alpha: 0.8),
                          blurRadius: 6,
                          spreadRadius: 1,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    title.toUpperCase(),
                    style: TextStyle(
                      color: accentColor,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2,
                      fontFamily: 'monospace',
                    ),
                  ),
                ],
              ),
              // Channel Indicator
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: _getBandColor(activeBand).withValues(alpha: 0.18),
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(
                    color: _getBandColor(activeBand).withValues(alpha: 0.6),
                    width: 1,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'BAND ${activeBand + 1} / $numBands',
                      style: TextStyle(
                        color: _getBandColor(activeBand),
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                        fontFamily: 'monospace',
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      '(${activeFreq.toStringAsFixed(0)} MHz)',
                      style: TextStyle(
                        color: Colors.grey[400],
                        fontSize: 10,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 14),

          // Intercept Status Banner
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: isScanning
                  ? (isHit
                      ? const Color(0xFF00E676).withValues(alpha: 0.12)
                      : const Color(0xFFFF1744).withValues(alpha: 0.10))
                  : Colors.transparent,
              borderRadius: BorderRadius.circular(6),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Icon(
                      isScanning
                          ? (isHit ? Icons.radar : Icons.sensors_off)
                          : Icons.radio,
                      size: 16,
                      color: isScanning
                          ? (isHit
                              ? const Color(0xFF00E676)
                              : const Color(0xFFFF1744))
                          : Colors.grey[500],
                    ),
                    const SizedBox(width: 8),
                    Text(
                      !isScanning
                          ? 'STANDBY // READY TO SCAN'
                          : (isHit
                              ? 'RADAR EMITTER INTERCEPTED (HIT)'
                              : 'SPECTRUM QUIESCENT (MISS)'),
                      style: TextStyle(
                        color: isScanning
                            ? (isHit
                                ? const Color(0xFF00E676)
                                : const Color(0xFFFF1744))
                            : Colors.grey[500],
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.8,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ),
                Text(
                  isScanning && isHit ? '+10.0 PTS' : '-1.0 PT',
                  style: TextStyle(
                    color: isScanning
                        ? (isHit
                            ? const Color(0xFF00E676)
                            : const Color(0xFFFF1744))
                        : Colors.transparent,
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    fontFamily: 'monospace',
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // 20 Vertical Spectrum Bars
          SizedBox(
            height: 180,
            child: LayoutBuilder(
              builder: (context, constraints) {
                final barWidth =
                    (constraints.maxWidth - (numBands - 1) * 4) / numBands;

                return Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: List.generate(numBands, (index) {
                    final isCurrent = index == activeBand;
                    final bandColor = _getBandColor(index);
                    final shadow = _getBandShadow(index);
                    final heightFactor = _getBandHeightFraction(index);

                    return Container(
                      width: barWidth,
                      margin: EdgeInsets.only(
                        right: index < numBands - 1 ? 4 : 0,
                      ),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.end,
                        children: [
                          // Animated Bar
                          AnimatedContainer(
                            duration: const Duration(milliseconds: 90),
                            curve: Curves.easeOutQuad,
                            height: (constraints.maxHeight - 20) * heightFactor,
                            decoration: BoxDecoration(
                              color: bandColor,
                              borderRadius: const BorderRadius.vertical(
                                top: Radius.circular(3),
                              ),
                              boxShadow: shadow,
                              gradient: isCurrent
                                  ? LinearGradient(
                                      begin: Alignment.topCenter,
                                      end: Alignment.bottomCenter,
                                      colors: [
                                        bandColor,
                                        bandColor.withValues(alpha: 0.6),
                                      ],
                                    )
                                  : null,
                            ),
                          ),
                          const SizedBox(height: 6),
                          // Band Index
                          Text(
                            '${index + 1}',
                            style: TextStyle(
                              color: isCurrent ? bandColor : Colors.grey[600],
                              fontSize: 9,
                              fontWeight: isCurrent
                                  ? FontWeight.bold
                                  : FontWeight.normal,
                              fontFamily: 'monospace',
                            ),
                          ),
                        ],
                      ),
                    );
                  }),
                );
              },
            ),
          ),

          const SizedBox(height: 8),

          // Spectrum Frequency Range Markers
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '${(minFreqMhz / 1000).toStringAsFixed(2)} GHz',
                style: TextStyle(
                  color: Colors.grey[600],
                  fontSize: 10,
                  fontFamily: 'monospace',
                ),
              ),
              Text(
                'RF SPECTRUM FREQUENCY BANDS (N=20)',
                style: TextStyle(
                  color: Colors.grey[600],
                  fontSize: 9,
                  letterSpacing: 1.0,
                  fontFamily: 'monospace',
                ),
              ),
              Text(
                '${(maxFreqMhz / 1000).toStringAsFixed(2)} GHz',
                style: TextStyle(
                  color: Colors.grey[600],
                  fontSize: 10,
                  fontFamily: 'monospace',
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
