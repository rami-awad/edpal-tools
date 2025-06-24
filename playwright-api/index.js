/**
 * Targeted Notification Automation Service
 * 
 * This service provides an API endpoint (/launch) that:
 * 1. Authenticates with an admin portal using credentials from environment variables
 * 2. Fills out a targeted notification form with:
 *    - Static age range filters from environment variables
 *    - Dynamic notification content (titles/bodies in English/Arabic) from POST request
 * 3. Submits the form and verifies the expected number of matching recipients
 * 4. Confirms and sends the notification
 * 5. Validates successful delivery
 * 
 * Uses Playwright for browser automation and Express for the API server.
 * Environment variables required: BASE_URL, USERNAME, PASSWORD, AGE_MIN, AGE_MAX, EXPECTED_ROW_COUNT
 */

require('dotenv').config();
const express = require('express');
const { chromium } = require('playwright');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;

app.post('/launch', async (req, res) => {
  const {
    filters_notification_title_en,
    filters_notification_body_en,
    filters_notification_title_ar,
    filters_notification_body_ar,
    filters_in_app_notification_title_en,
    markdown_editor_in_app_notification_body_en,
    filters_in_app_notification_title_ar,
    markdown_editor_in_app_notification_body_ar,
  } = req.body;

  const {
    BASE_URL,
    USERNAME,
    PASSWORD,
    AGE_MIN,
    AGE_MAX,
    EXPECTED_ROW_COUNT,
  } = process.env;

  const fullUrl = `${BASE_URL}/admin/targeted_notification/targeted_notifications`;

  const urlWithAuth = new URL(fullUrl);
  urlWithAuth.username = USERNAME;
  urlWithAuth.password = PASSWORD;

  const browser = await chromium.launch({
    headless: process.env.HEADLESS !== 'false'  // Defaults to true unless explicitly "false"
  });
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    console.log(`Navigating to admin targeted notification page`);
    await page.goto(urlWithAuth.href, { waitUntil: 'networkidle' });

    // Fill static age fields from env
    console.log(`Filling AGE_MIN: ${AGE_MIN}`);
    await page.fill('#filters_student_age_between_min', AGE_MIN);

    console.log(`Filling AGE_MAX: ${AGE_MAX}`);
    await page.fill('#filters_student_age_between_max', AGE_MAX);

    // Fill dynamic fields from POST body
    console.log(`Filling filters_notification_title_en: ${filters_notification_title_en}`);
    await page.fill('#filters_notification_title_en', filters_notification_title_en);

    console.log(`Filling filters_notification_body_en: ${filters_notification_body_en}`);
    await page.fill('#filters_notification_body_en', filters_notification_body_en);

    console.log(`Filling filters_notification_title_ar: ${filters_notification_title_ar}`);
    await page.fill('#filters_notification_title_ar', filters_notification_title_ar);

    console.log(`Filling filters_notification_body_ar: ${filters_notification_body_ar}`);
    await page.fill('#filters_notification_body_ar', filters_notification_body_ar);

    console.log(`Filling filters_in_app_notification_title_en: ${filters_in_app_notification_title_en}`);
    await page.fill('#filters_in_app_notification_title_en', filters_in_app_notification_title_en);

    console.log(`Filling markdown_editor_in_app_notification_body_en: ${markdown_editor_in_app_notification_body_en}`);
    await page.fill('#markdown_editor_in_app_notification_body_en', markdown_editor_in_app_notification_body_en);

    console.log(`Filling filters_in_app_notification_title_ar: ${filters_in_app_notification_title_ar}`);
    await page.fill('#filters_in_app_notification_title_ar', filters_in_app_notification_title_ar);

    console.log(`Filling markdown_editor_in_app_notification_body_ar: ${markdown_editor_in_app_notification_body_ar}`);
    await page.fill('#markdown_editor_in_app_notification_body_ar', markdown_editor_in_app_notification_body_ar);

    console.log('Filled all required form fields. Submitting filter form.');

    // Click Filter button
    console.log('Clicking "Filter" button');
    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle' }),
      page.click('button[name="_save"]:has-text("Filter")'),
    ]);

    console.log('Waiting for table to appear...');

    // Wait for table and check rows
    const tableSelector = 'table.table.table-condensed.table-striped.table-hover';
    await page.waitForSelector(tableSelector);

    const rowCount = await page.$$eval(`${tableSelector} tbody tr`, rows => rows.length);
    console.log(`Table loaded. Row count: ${rowCount}`);

    if (rowCount < parseInt(EXPECTED_ROW_COUNT)) {
      throw new Error(`Expected at least ${EXPECTED_ROW_COUNT} rows, found ${rowCount}`);
    }

    console.log(`Found ${rowCount} rows. Proceeding to click "Confirm and send notification"`);

    // Click Confirm and send notification
    console.log('Clicking "Confirm and send notification" button');
    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle' }),
      page.click('button[name="_save"]:has-text("Confirm and send notification")'),
    ]);

    // Confirm success by checking for "Student Profile" heading
    console.log('Waiting for confirmation "Student Profile" heading');
    await page.waitForSelector('h3:has-text("Student Profile")');
    console.log('Notification sent successfully. Found "Student Profile" heading.');

    res.json({ status: 'Success', message: 'Notification sent successfully.' });

  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).json({ status: 'Error', message: error.message });
  } finally {
    console.log('Closing browser.');
    await browser.close();
  }
});

app.listen(PORT, () => {
  console.log(`Playwright server running on http://localhost:${PORT}`);
});