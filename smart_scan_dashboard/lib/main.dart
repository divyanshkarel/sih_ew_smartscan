import 'package:flutter/material.dart';
import 'screens/dashboard_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SmartScanDashboardApp());
}

class SmartScanDashboardApp extends StatelessWidget {
  const SmartScanDashboardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Electronic Warfare Smart Scan War Room',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.dark,
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF060911),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00E5FF), // Electric Cyan
          secondary: Color(0xFF00E676), // Neon Green
          surface: Color(0xFF0D1321),
          error: Color(0xFFFF1744),
        ),
        useMaterial3: true,
      ),
      home: const DashboardScreen(),
    );
  }
}
