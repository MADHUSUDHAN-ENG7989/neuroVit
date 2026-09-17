/**
 * setup-dynamo.js
 * Run once to create the DynamoDB tables: Users and QueryLogs
 */
require('dotenv').config();
const { DynamoDBClient, CreateTableCommand, ListTablesCommand } = require('@aws-sdk/client-dynamodb');

const client = new DynamoDBClient({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  },
});

async function createTable(params) {
  try {
    const res = await client.send(new CreateTableCommand(params));
    console.log(`✅ Created table: ${params.TableName}`);
    return res;
  } catch (err) {
    if (err.name === 'ResourceInUseException') {
      console.log(`⚠️  Table already exists: ${params.TableName}`);
    } else {
      console.error(`❌ Failed to create ${params.TableName}:`, err.message);
      throw err;
    }
  }
}

async function main() {
  console.log(`Connecting to DynamoDB in region: ${process.env.AWS_REGION}`);

  // 1. Users table
  await createTable({
    TableName: 'NeuroSight_Users',
    AttributeDefinitions: [{ AttributeName: 'email', AttributeType: 'S' }],
    KeySchema: [{ AttributeName: 'email', KeyType: 'HASH' }],
    BillingMode: 'PAY_PER_REQUEST',
  });

  // 2. QueryLogs table
  await createTable({
    TableName: 'NeuroSight_QueryLogs',
    AttributeDefinitions: [
      { AttributeName: 'logId', AttributeType: 'S' },
      { AttributeName: 'timestamp', AttributeType: 'S' },
    ],
    KeySchema: [
      { AttributeName: 'logId', KeyType: 'HASH' },
      { AttributeName: 'timestamp', KeyType: 'RANGE' },
    ],
    BillingMode: 'PAY_PER_REQUEST',
  });

  // Verify
  const { TableNames } = await client.send(new ListTablesCommand({}));
  console.log('\n📋 All tables in your account:', TableNames);
}

main().catch(console.error);
