import api from './axios';

/**
 * Centralized API service layer for the
 * AI Customer Retention Platform.
 *
 * Important:
 * - No mock/fallback business data is returned.
 * - Backend remains the single source of truth.
 * - API errors are logged and propagated to the UI.
 */

/* =========================================================
   HELPERS
   ========================================================= */

/**
 * Extract a readable error message from an Axios error.
 */
const getErrorMessage = (error, fallbackMessage) => {
  return (
    error?.response?.data?.detail ||
    error?.response?.data?.message ||
    error?.message ||
    fallbackMessage
  );
};

/**
 * Log API errors consistently.
 */
const handleApiError = (context, error) => {
  console.error(`${context}:`, {
    message: getErrorMessage(error, 'Unknown API error'),
    status: error?.response?.status,
    data: error?.response?.data,
  });
};


/* =========================================================
   ANALYTICS
   ========================================================= */

/**
 * Fetch real-time analytics from the backend.
 */
export const getAnalytics = async () => {
  try {
    const response = await api.get('/analytics');

    return response.data;
  } catch (error) {
    handleApiError(
      'Error fetching analytics',
      error
    );

    throw error;
  }
};


/**
 * Fetch dashboard metrics.
 *
 * The backend analytics endpoint is the
 * single source of truth.
 */
export const getMetrics = async () => {
  try {
    const data = await getAnalytics();

    return {
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
    };
  } catch (error) {
    handleApiError(
      'Error fetching metrics',
      error
    );

    throw error;
  }
};


/* =========================================================
   CUSTOMERS
   ========================================================= */

/**
 * Fetch all customers from the backend.
 */
export const getAllCustomers = async () => {
  try {
    const response = await api.get(
      '/customers'
    );

    const customers = response.data?.customers;

    return Array.isArray(customers)
      ? customers
      : [];
  } catch (error) {
    handleApiError(
      'Error fetching customers',
      error
    );

    throw error;
  }
};


/**
 * Fetch a single customer by database ID.
 */
export const getCustomerById = async (id) => {
  if (
    id === undefined ||
    id === null ||
    id === ''
  ) {
    throw new Error(
      'Customer ID is required.'
    );
  }

  try {
    const response = await api.get(
      `/customers/${encodeURIComponent(id)}`
    );

    return response.data;
  } catch (error) {
    handleApiError(
      'Error fetching customer',
      error
    );

    throw error;
  }
};


/**
 * Fetch prediction history for a customer.
 */
export const getCustomerPredictions = async (
  id
) => {
  if (
    id === undefined ||
    id === null ||
    id === ''
  ) {
    throw new Error(
      'Customer ID is required.'
    );
  }

  try {
    const response = await api.get(
      `/customers/${encodeURIComponent(id)}/predictions`
    );

    const predictions =
      response.data?.predictions;

    return Array.isArray(predictions)
      ? predictions
      : [];
  } catch (error) {
    handleApiError(
      'Error fetching customer predictions',
      error
    );

    throw error;
  }
};


/* =========================================================
   PREDICTION
   ========================================================= */

/**
 * Predict churn for a single customer.
 */
export const predictChurn = async (
  customerData
) => {
  if (
    !customerData ||
    typeof customerData !== 'object'
  ) {
    throw new Error(
      'Valid customer data is required.'
    );
  }

  try {
    const response = await api.post(
      '/predict',
      customerData
    );

    return response.data;
  } catch (error) {
    handleApiError(
      'Error predicting churn',
      error
    );

    throw error;
  }
};


/**
 * Predict churn and save the result
 * to the database.
 */
export const predictAndSave = async (
  customerData,
  customerName = '',
  customerEmail = ''
) => {
  if (
    !customerData ||
    typeof customerData !== 'object'
  ) {
    throw new Error(
      'Valid customer data is required.'
    );
  }

  try {
    const response = await api.post(
      '/predict-and-save',
      customerData,
      {
        params: {
          customer_name: customerName,
          customer_email: customerEmail,
        },
      }
    );

    return response.data;
  } catch (error) {
    handleApiError(
      'Error predicting and saving customer',
      error
    );

    throw error;
  }
};


/* =========================================================
   HIGH-RISK CUSTOMERS
   ========================================================= */

/**
 * Parse prediction reasons safely.
 *
 * The backend may return reasons as:
 * - JSON string
 * - Array
 * - null
 */
const parseReasons = (reasons) => {
  if (Array.isArray(reasons)) {
    return reasons;
  }

  if (
    typeof reasons === 'string' &&
    reasons.trim()
  ) {
    try {
      const parsed = JSON.parse(
        reasons
      );

      if (Array.isArray(parsed)) {
        return parsed;
      }

      return [reasons];
    } catch {
      return [reasons];
    }
  }

  return [];
};


/**
 * Fetch top high-risk customers.
 *
 * NOTE:
 * This currently uses the existing backend APIs.
 * A dedicated /customers/high-risk endpoint
 * can be introduced later to make this operation
 * more efficient.
 */
export const getHighRiskCustomers = async () => {
  try {
    const customers =
      await getAllCustomers();

    if (!customers.length) {
      return [];
    }

    const highRiskList = [];

    /*
     * We intentionally limit the number of
     * prediction-history requests for now
     * because the current backend does not
     * provide a bulk high-risk endpoint.
     *
     * This is NOT mock data.
     */
    const customersToCheck =
      customers.slice(0, 50);

    const predictionResults =
      await Promise.allSettled(
        customersToCheck.map(
          async (customer) => {
            const predictions =
              await getCustomerPredictions(
                customer.id
              );

            return {
              customer,
              predictions,
            };
          }
        )
      );

    for (
      const result of predictionResults
    ) {
      if (
        result.status !== 'fulfilled'
      ) {
        continue;
      }

      const {
        customer,
        predictions,
      } = result.value;

      if (!predictions.length) {
        continue;
      }

      /*
       * Backend already returns prediction
       * history ordered newest first.
       */
      const latest =
        predictions[0];

      let risk =
        Number(
          latest?.churn_probability ?? 0
        );

      /*
       * Support both:
       * 0.87
       * and
       * 87
       */
      if (
        risk > 0 &&
        risk <= 1
      ) {
        risk *= 100;
      }

      if (risk < 60) {
        continue;
      }

      const reasons =
        parseReasons(
          latest?.reasons
        );

      highRiskList.push({
        id: customer.id,

        name:
          customer.name ||
          'Unknown Customer',

        email:
          customer.email ||
          '',

        risk: Math.round(
          Math.min(
            Math.max(risk, 0),
            100
          )
        ),

        riskLevel:
          latest?.risk_level ||
          (
            risk >= 80
              ? 'Critical Risk'
              : 'High Risk'
          ),

        reason:
          reasons[0] ||
          'AI detected elevated churn risk.',
      });
    }

    /*
     * Highest risk first.
     */
    highRiskList.sort(
      (a, b) =>
        b.risk - a.risk
    );

    return highRiskList.slice(
      0,
      5
    );

  } catch (error) {
    handleApiError(
      'Error fetching high-risk customers',
      error
    );

    throw error;
  }
};
