import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  getMetrics,
  getHighRiskCustomers,
  predictChurn,
} from '../api/services';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts';

const Dashboard = () => {
  const [metrics, setMetrics] = useState({
    totalCustomers: 0,
    activeCustomers: 0,
    highRisk: 0,
    criticalRisk: 0,
    retentionRate: 0,
    revenue: 0,
    revenueLoss: 0,
    predictionsCount: 0,
    churnTrend: [],
  });

  const [topRisky, setTopRisky] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [riskWarning, setRiskWarning] = useState(null);

  const [testCustomer, setTestCustomer] = useState({
    gender: 'Male',
    SeniorCitizen: 0,
    Partner: 'Yes',
    Dependents: 'No',
    tenure: 12,
    PhoneService: 'Yes',
    InternetService: 'Fiber optic',
    Contract: 'Month-to-month',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 85.5,
    TotalCharges: 1020.5,
  });

  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);

  /* =====================================================
     LOAD DASHBOARD DATA
     ===================================================== */

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      setRiskWarning(null);

      /*
       * Use Promise.allSettled so that failure of the
       * high-risk section does not break the entire dashboard.
       */
      const results = await Promise.allSettled([
        getMetrics(),
        getHighRiskCustomers(),
      ]);

      const [metricsResult, riskyResult] = results;

      /* ---------------- Analytics ---------------- */

      if (metricsResult.status === 'fulfilled') {
        const data = metricsResult.value;

        setMetrics({
          totalCustomers: Number(
            data?.totalCustomers ?? 0
          ),

          activeCustomers: Number(
            data?.activeCustomers ?? 0
          ),

          highRisk: Number(
            data?.highRisk ?? 0
          ),

          criticalRisk: Number(
            data?.criticalRisk ?? 0
          ),

          retentionRate: Number(
            data?.retentionRate ?? 0
          ),

          revenue: Number(
            data?.revenue ?? 0
          ),

          revenueLoss: Number(
            data?.revenueLoss ?? 0
          ),

          predictionsCount: Number(
            data?.predictionsCount ?? 0
          ),

          churnTrend: Array.isArray(
            data?.churnTrend
          )
            ? data.churnTrend
            : [],
        });
      } else {
        console.error(
          'Dashboard analytics error:',
          metricsResult.reason
        );

        setError(
          'Failed to load dashboard analytics.'
        );
      }

      /* ---------------- High Risk ---------------- */

      if (riskyResult.status === 'fulfilled') {
        setTopRisky(
          Array.isArray(riskyResult.value)
            ? riskyResult.value
            : []
        );
      } else {
        console.error(
          'High-risk customers error:',
          riskyResult.reason
        );

        /*
         * Dashboard remains usable even if the
         * high-risk section is temporarily unavailable.
         */
        setTopRisky([]);

        setRiskWarning(
          'High-risk customer data is temporarily unavailable.'
        );
      }

      setLoading(false);
    };

    fetchData();
  }, []);

  /* =====================================================
     AI PREDICTION TESTER
     ===================================================== */

  const handlePredict = async () => {
    try {
      setPredicting(true);
      setPredictionResult(null);

      const response =
        await predictChurn(testCustomer);

      setPredictionResult(response);
    } catch (error) {
      console.error(
        'Prediction error:',
        error
      );

      const message =
        error?.response?.data?.detail ||
        error?.response?.data?.message ||
        error?.message ||
        'Failed to get churn prediction.';

      alert(message);
    } finally {
      setPredicting(false);
    }
  };

  /* =====================================================
     LOADING STATE
     ===================================================== */

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">
          Loading dashboard...
        </div>
      </div>
    );
  }

  /* =====================================================
     CHURN TREND
     ===================================================== */

  const churnData =
    Array.isArray(metrics.churnTrend)
      ? metrics.churnTrend
      : [];

  /* =====================================================
     RENDER
     ===================================================== */

  return (
    <div className="p-6 max-w-7xl mx-auto">

      {/* =================================================
          ANALYTICS ERROR
          ================================================= */}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 p-4 mb-6 rounded-lg">
          <p className="font-semibold">
            ❌ {error}
          </p>

          <p className="text-sm mt-1">
            Please check the backend analytics service.
          </p>
        </div>
      )}

      {/* =================================================
          HIGH-RISK WARNING
          ================================================= */}

      {riskWarning && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 p-4 mb-6 rounded-lg">
          <p className="font-semibold">
            ⚠️ High-Risk Data Warning
          </p>

          <p className="text-sm mt-1">
            {riskWarning}
          </p>
        </div>
      )}

      {/* =================================================
          CRITICAL RISK ALERT
          ================================================= */}

      {topRisky.some(
        (customer) =>
          Number(customer.risk) >= 80
      ) && (
        <div className="bg-red-50 border-l-4 border-red-600 p-4 mb-6 rounded">
          <div className="flex items-center">
            <span className="text-2xl mr-3">
              🚨
            </span>

            <div>
              <p className="font-bold text-red-700">
                Critical Risk Alert!
              </p>

              <p className="text-sm text-red-600">
                {
                  topRisky.filter(
                    (customer) =>
                      Number(customer.risk) >= 80
                  ).length
                }{' '}
                customers are at critical risk
                (80%+ churn probability).
                Immediate action recommended.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* =================================================
          METRICS CARDS
          ================================================= */}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">

        {/* Total Customers */}
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-blue-500">
          <p className="text-gray-500 text-sm">
            Total Customers
          </p>

          <p className="text-2xl font-bold">
            {Number(
              metrics.totalCustomers
            ).toLocaleString()}
          </p>
        </div>

        {/* Active Customers */}
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-green-500">
          <p className="text-gray-500 text-sm">
            Active Customers
          </p>

          <p className="text-2xl font-bold text-green-600">
            {Number(
              metrics.activeCustomers
            ).toLocaleString()}
          </p>
        </div>

        {/* High Risk */}
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-red-500">
          <p className="text-gray-500 text-sm">
            High Risk
          </p>

          <p className="text-2xl font-bold text-red-600">
            {Number(
              metrics.highRisk
            ).toLocaleString()}
          </p>
        </div>

        {/* Retention Rate */}
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-purple-500">
          <p className="text-gray-500 text-sm">
            Retention Rate
          </p>

          <p className="text-2xl font-bold text-purple-600">
            {Number(
              metrics.retentionRate
            ).toFixed(1)}
            %
          </p>
        </div>

        {/* Revenue Loss */}
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-orange-500">
          <p className="text-gray-500 text-sm">
            Revenue Loss (Est.)
          </p>

          <p className="text-2xl font-bold text-orange-600">
            ₹
            {Number(
              metrics.revenueLoss
            ).toLocaleString()}
          </p>
        </div>
      </div>

      {/* =================================================
          AI PREDICTION TESTER
          ================================================= */}

      <div className="bg-white p-4 rounded-lg shadow-md mb-6">

        <h2 className="font-semibold text-lg mb-4">
          🧪 Test AI Prediction
        </h2>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">

          {/* Tenure */}
          <input
            className="border rounded p-2 text-sm"
            placeholder="Tenure"
            type="number"
            min="0"
            value={testCustomer.tenure}
            onChange={(e) =>
              setTestCustomer({
                ...testCustomer,
                tenure:
                  parseInt(
                    e.target.value,
                    10
                  ) || 0,
              })
            }
          />

          {/* Monthly Charges */}
          <input
            className="border rounded p-2 text-sm"
            placeholder="Monthly Charges"
            type="number"
            min="0"
            step="0.01"
            value={
              testCustomer.MonthlyCharges
            }
            onChange={(e) =>
              setTestCustomer({
                ...testCustomer,
                MonthlyCharges:
                  parseFloat(
                    e.target.value
                  ) || 0,
              })
            }
          />

          {/* Contract */}
          <select
            className="border rounded p-2 text-sm"
            value={
              testCustomer.Contract
            }
            onChange={(e) =>
              setTestCustomer({
                ...testCustomer,
                Contract:
                  e.target.value,
              })
            }
          >
            <option value="Month-to-month">
              Month-to-month
            </option>

            <option value="One year">
              One year
            </option>

            <option value="Two year">
              Two year
            </option>
          </select>

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={predicting}
            className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 transition disabled:opacity-50"
          >
            {predicting
              ? 'Predicting...'
              : '🔮 Predict Churn'}
          </button>
        </div>

        {/* Prediction Result */}
        {predictionResult && (
          <div
            className={`p-4 rounded ${
              predictionResult.risk_level ===
              'Low Risk'
                ? 'bg-green-50'
                : predictionResult.risk_level ===
                  'Medium Risk'
                ? 'bg-yellow-50'
                : 'bg-red-50'
            }`}
          >
            <p>
              <strong>
                Probability:
              </strong>{' '}
              {Number(
                predictionResult.probability ??
                  predictionResult.churn_probability ??
                  0
              ).toFixed(1)}
              %
            </p>

            <p>
              <strong>
                Risk Level:
              </strong>{' '}
              <span className="font-bold">
                {predictionResult.risk_color ||
                  '⚠️'}{' '}
                {predictionResult.risk_level ||
                  'Unknown'}
              </span>
            </p>

            <p>
              <strong>
                Reasons:
              </strong>{' '}
              {Array.isArray(
                predictionResult.reasons
              )
                ? predictionResult.reasons.join(
                    ', '
                  )
                : predictionResult.reasons ||
                  'No specific reasons available.'}
            </p>

            <p>
              <strong>
                Recommendation:
              </strong>{' '}
              {predictionResult.recommendation ||
                'No recommendation available.'}
            </p>
          </div>
        )}
      </div>

      {/* =================================================
          MONTHLY CHURN TREND
          ================================================= */}

      <div className="bg-white p-4 rounded-lg shadow-md mb-6">

        <h2 className="font-semibold text-lg mb-4">
          📈 Monthly Churn Trend
        </h2>

        {churnData.length === 0 ? (
          <div className="py-10 text-center">
            <p className="text-gray-500">
              No churn trend data available yet.
            </p>

            <p className="text-gray-400 text-sm mt-1">
              Historical prediction data will appear
              here as the platform collects more results.
            </p>
          </div>
        ) : (
          <ResponsiveContainer
            width="100%"
            height={250}
          >
            <BarChart data={churnData}>
              <CartesianGrid
                strokeDasharray="3 3"
              />

              <XAxis dataKey="month" />

              <YAxis />

              <Tooltip />

              <Bar
                dataKey="churn"
                fill="#f97316"
              />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* =================================================
          TOP HIGH-RISK CUSTOMERS
          ================================================= */}

      <div className="bg-white p-4 rounded-lg shadow-md">

        <h2 className="font-semibold text-lg mb-4">
          🔥 Top High Risk Customers
        </h2>

        {topRisky.length === 0 ? (
          <div className="py-8 text-center">
            <p className="text-gray-500">
              No high-risk customers detected.
            </p>

            <p className="text-gray-400 text-sm mt-1">
              Customers will appear here after
              churn predictions are generated.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">

            <table className="w-full text-left">

              <thead className="bg-gray-50">
                <tr>
                  <th className="p-3 text-sm font-semibold text-gray-600">
                    Name
                  </th>

                  <th className="p-3 text-sm font-semibold text-gray-600">
                    Email
                  </th>

                  <th className="p-3 text-sm font-semibold text-gray-600">
                    Risk
                  </th>

                  <th className="p-3 text-sm font-semibold text-gray-600">
                    Reason
                  </th>
                </tr>
              </thead>

              <tbody>
                {topRisky.map(
                  (customer) => {
                    const risk =
                      Number(
                        customer.risk
                      ) || 0;

                    return (
                      <tr
                        key={customer.id}
                        className="border-t hover:bg-gray-50 transition"
                      >
                        <td className="p-3 font-medium">
                          <Link
                            to={`/customer/${customer.id}`}
                            className="text-indigo-600 hover:underline"
                          >
                            {customer.name ||
                              'Unknown Customer'}
                          </Link>
                        </td>

                        <td className="p-3 text-gray-600">
                          {customer.email ||
                            '—'}
                        </td>

                        <td className="p-3">
                          <span
                            className={`px-3 py-1 rounded-full text-white text-sm font-medium ${
                              risk >= 80
                                ? 'bg-red-600'
                                : risk >= 60
                                ? 'bg-orange-500'
                                : 'bg-yellow-500'
                            }`}
                          >
                            {Math.round(
                              risk
                            )}
                            %
                          </span>
                        </td>

                        <td className="p-3 text-gray-600">
                          {customer.reason ||
                            'AI detected elevated churn risk.'}
                        </td>
                      </tr>
                    );
                  }
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
