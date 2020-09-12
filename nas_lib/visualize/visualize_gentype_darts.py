from graphviz import Digraph
import pickle


def plot(genotype, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot'
    )
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=True)


def load_model(file_path):
    with open(file_path, 'rb') as f:
        gtyp = pickle.load(f)
    return gtyp


if __name__ == '__main__':
    model_name = 'npubo'     # options: ['npubo_type2', 'npubo_type3', 'npenas_type3']
    # NPUBO-Type3
    # model_path = '/home/albert_wei/Desktop/nas_models/titanxp_2/da6e5c2b7f6c8f09435559d707add7a6715b9f578743610c363bb69f9e8c8dd5.pkl'
    # NPUBO - Type2
    # model_path = '/home/albert_wei/Desktop/nas_models/liang_titan_V/97de0acd0ab4b320fb2b774118527357dac65142c1fee8a47595c7153db5b705.pkl'
    # NPENAS-Type3
    # model_path = '/home/albert_wei/Desktop/nas_models/2080ti/models/a945af02dc233989a6192d7e3462ea8e9c2a49764a55854e816b95a830ec9875.pkl'
    model_path = '/home/albert_wei/Desktop/NPENAS_materials/论文资料/results_models/npubo_150_seed_4/4f4ab32092cd83f30d3ac249e88c0828a870df8db86282e651eaef4d0b1f397a.pkl'
    genotype = load_model(model_path)

    plot(genotype.normal, f"normal_{model_name}")
    plot(genotype.reduce, f"reduction_{model_name}")

