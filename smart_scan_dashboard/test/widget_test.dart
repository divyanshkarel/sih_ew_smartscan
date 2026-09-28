import 'package:flutter_test/flutter_test.dart';
import 'package:smart_scan_dashboard/main.dart';

void main() {
  testWidgets('SmartScanDashboardApp renders title and structure', (WidgetTester tester) async {
    await tester.pumpWidget(const SmartScanDashboardApp());

    // Verify app renders with the command header
    expect(find.text('ELECTRONIC WARFARE COMMAND // SMART SCAN WAR ROOM'), findsOneWidget);
  });
}
