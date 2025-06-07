#!/bin/bash

set -e  # Зупиняємо скрипт при будь-якій помилці

echo "=== 0. Cleaning up any existing cluster ==="
# Вбиваємо будь-які існуючі переадресації портів
pkill -f "kubectl port-forward.*raycluster-kuberay-head-svc" || true

# Видаляємо існуючий кластер, якщо він існує
kind delete cluster --name ray-cluster || true

echo "=== 1. Starting Kind cluster ==="
mkdir -p /tmp/kubeflow-data
kind create cluster --config kind/kind-config.yaml

# Очікуємо готовності кластера
echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=ready nodes --all --timeout=300s

echo "=== 2. Installing KubeRay operator ==="
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Встановлюємо оператор з покращеними налаштуваннями для стабільності
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.2 


echo "=== 3. Waiting for KubeRay operator to be ready ==="
kubectl wait --for=condition=available --timeout=300s deployment/kuberay-operator

# Перевіряємо статус оператора
echo "Checking operator status..."
kubectl get deployment kuberay-operator -o wide

echo "=== 4. Installing Ray cluster with SCALE-TO-ZERO autoscaling ==="
helm install raycluster kuberay/ray-cluster --version 1.3.2 -f ray-cluster-values.yaml

echo "=== 5. Waiting for Ray cluster to be ready ==="

# Проста функція очікування створення ресурсу
wait_for_resource_to_exist() {
    local resource_type=$1
    local selector=$2
    local description=$3
    
    echo "Waiting for $description to be created..."
    while true; do
        if kubectl get $resource_type -l $selector --no-headers 2>/dev/null | grep -q .; then
            echo "✅ $description found"
            break
        fi
        echo "⏳ $description not found yet, waiting..."
        sleep 10
    done
}

# Чекаємо створення head pod
wait_for_resource_to_exist "pod" "ray.io/node-type=head" "head pod"

# Тепер безпечно чекаємо готовності head pod
echo "Waiting for head pod to be ready..."
kubectl wait --for=condition=ready --timeout=600s pod -l ray.io/node-type=head

echo "=== 6. Verifying cluster health ==="
# Перевіряємо статус кластера
echo "Checking cluster status..."
kubectl get pods -l ray.io/cluster-name -o wide

# Перевіряємо статус Ray всередині кластера
echo "Checking Ray status..."
HEAD_POD=$(kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
if [ ! -z "$HEAD_POD" ]; then
    echo "Head pod found: $HEAD_POD"
    
    # Проста перевірка Ray статусу з кількома спробами
    echo "Waiting for Ray to be ready inside head pod..."
    for i in {1..10}; do
        if kubectl exec $HEAD_POD -- ray status 2>/dev/null; then
            echo "✅ Ray is ready!"
            break
        fi
        echo "⏳ Ray not ready yet, attempt $i/10..."
        sleep 15
    done
else
    echo "❌ Head pod not found"
    exit 1
fi

echo "=== 7. Setting up port forwarding ==="
# Вбиваємо будь-які існуючі переадресації портів
pkill -f "kubectl port-forward.*raycluster-kuberay-head-svc" || true

# Налаштовуємо переадресації портів у фоновому режимі
echo "Starting port forwards..."
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265 > /dev/null 2>&1 &
kubectl port-forward service/raycluster-kuberay-head-svc 10001:10001 > /dev/null 2>&1 &
kubectl port-forward service/raycluster-kuberay-head-svc 8000:8000 > /dev/null 2>&1 &

# Чекаємо встановлення переадресацій портів
echo "Waiting for port forwards to establish..."
sleep 10

# Проста перевірка доступності
echo "Checking service availability..."
curl -s --connect-timeout 3 http://localhost:8265 > /dev/null && echo "✅ Dashboard accessible" || echo "⚠️  Dashboard may need more time to start"

echo "=== Ray cluster setup completed! ==="
echo ""
echo "🎉 Ray Dashboard available at:"
echo "   http://localhost:8265"
echo ""
echo "🔗 Ray Client connection:"
echo "   ray.init('ray://localhost:10001')"
echo ""
echo "🌐 Ray Serve endpoint:"
echo "   http://localhost:8000"
echo ""
echo "📊 To check cluster status:"
echo "   kubectl get pods"
echo "   kubectl get services"
echo "   kubectl exec \$(kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}') -- ray status"
echo ""
echo "🧪 To monitor pod stability:"
echo "   watch kubectl get pods"
echo ""
echo "🔧 To check autoscaler status:"
echo "   kubectl logs \$(kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}') -c autoscaler"
echo ""
echo "⚠️  Note: Port forwards are running in background. To stop them:"
echo "   pkill -f 'kubectl port-forward.*raycluster-kuberay-head-svc'"
echo ""
echo "🚀 SCALE-TO-ZERO ENABLED:"
echo "   - Worker pods: 0 → 10 (created on demand)"
echo "   - Idle timeout: 60 seconds" 
echo "   - Submit jobs to automatically create workers!"
echo ""
echo "🎯 Test scale-to-zero with:"
echo "   python validate_ray_job.py minimal_ray_job.py"
echo "   ray job submit --address=http://localhost:8265 -- python minimal_ray_job.py"
