{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -Wno-orphans #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE RankNTypes #-}

module Data.Accelerate.TensorFlow.Desugar where

import Data.Array.Accelerate.AST.Operation
    ( expType,
      paramsIn',
      fromGrounds,
      PrimFun(..),
      PreArgs((:>:), ArgsNil),
      ExpVars,
      PreOpenAcc(..),
      GroundR(GroundRscalar),
      Modifier(In, Mut, Out),
      PreOpenExp(..),
      PreOpenFun(Body, Lam),
      ArrayInstr(..),
      GroundVars,
      OperationAcc,
      Out,
      In,
      Mut,
      Arg(ArgFun, ArgArray, ArgVar), Var (..), PrimConst (..), Fun', PrimMaybe )
import Data.Array.Accelerate.AST.LeftHandSide
    ( LeftHandSide(LeftHandSideWildcard, LeftHandSideSingle) )
import Data.Array.Accelerate.Backend
    ( DesugarAcc(mkGenerate, mkPermute, mkMap) )
import Data.Array.Accelerate.Type
    ( Word8,
      scalarTypeWord8,
      scalarTypeInt,
      IntegralType(..),
      NumType(IntegralNumType, FloatingNumType),
      SingleType(NumSingleType),
      ScalarType(SingleScalarType), BoundedType (..) )
import Data.Array.Accelerate.Representation.Array
    ( Buffer, Buffers, ArrayR(ArrayR) )
import Data.Array.Accelerate.Representation.Type
    ( Distributes(reprIsSingle),
      Distribute,
      TupR(TupRpair, TupRunit, TupRsingle),
      TypeR )
import Data.Array.Accelerate.Trafo.Var
    ( declareVars, DeclareVars(DeclareVars) )
import Data.Array.Accelerate.AST.Idx ( Idx(ZeroIdx, SuccIdx) )
import Data.Array.Accelerate.Trafo.Operation.Substitution
    ( weakenArrayInstr, aletUnique, Sink(..) )
import Data.Array.Accelerate.AST.Environment
    ( mapEnv, prj', (.>), weakenId, push', type (:>), Env(Empty) )
import Data.Array.Accelerate.Representation.Ground ( buffersR )
import Data.Array.Accelerate.Trafo.Desugar ( desugarAlloc )
import Data.Array.Accelerate.Trafo.Exp.Substitution
    ( weakenVars, SinkExp(weakenE), rebuildNoArrayInstr )
import Data.Type.Equality ( type (:~:)(Refl) )
import Data.Array.Accelerate.Representation.Shape
    ( dim1, shapeType, ShapeR(..), DIM1 )
import Prelude hiding (exp)

import Data.Accelerate.TensorFlow.Type
    ( TensorTypeDict(TensorTypeDict),
      tfTensorTypeDict,
      tfAllDict,
      tfFloatDict,
      tfIntDict,
      tfModDict,
      tfNum'Dict,
      tfNumDict,
      tfOrdDict,
      OneOfDict(OneOfDict), zero )
import Data.Accelerate.TensorFlow.Operation
    ( TensorOp(TWhere, TTensorScatter, TCast, TBooleanMask, TConstant,
               TId, TGather, TVar, TSelect, TAdd, TMul, TSub, TNeg, TAbs, TSign,
               TTruncateDiv, TTruncateMod, TBitwiseAnd, TBitwiseOr, TBitwiseXor,
               TInvert, TRealDiv, TReciprocal, TSin, TCos, TTan, TAsin, TAcos,
               TAtan, TSinh, TCosh, TTanh, TAsinh, TAcosh, TAtanh, TExp, TSqrt,
               TLog, TPow, TAtan2, TLess, TGreater, TLessEqual, TGreaterEqual,
               TEqual, TNotEqual, TMaximum, TMinimum, TLogicalAnd, TLogicalOr,
               TLogicalNot),
      ScatterFun(ScatterFunMin, ScatterFunAdd) )
import Data.Array.Accelerate.Interpreter (evalMinBound, evalPi, evalMaxBound)
import Data.Array.Accelerate.Representation.Slice
    ( SliceIndex(..), sliceDomainR, sliceEltR, sliceShapeR )

newtype BufferIdx benv a = BIdx (Idx benv (Buffer a))

instance Sink BufferIdx where
  weaken :: (env :> env') -> forall t. BufferIdx env t -> BufferIdx env' t
  weaken w (BIdx idx) = BIdx (weaken w idx)

type BufferEnv benv env = Env (BufferIdx benv) env

instance DesugarAcc TensorOp where
  mkMap :: Arg env (Fun' (s -> t)) -> Arg env (In sh s) -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
  mkMap (ArgFun (Lam lhs (Body body))) (ArgArray _ (ArrayR _ t) _ gvb) aOut =
    mkExp (push' Empty (lhs, distributeBIdx t gvb)) body aOut
  mkMap _ _ _ = error "impossible"

  mkGenerate :: Arg env (Fun' (sh -> t)) -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
  mkGenerate f (ArgArray _ (ArrayR sh t) gv gvb)
    | DeclareVars lhs w k          <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
    , DeclareVars lhs' w' k'       <- declareVars $ buffersR (TupRsingle scalarTypeInt)
    , DeclareVars lhs'' w'' k''    <- declareVars $ buffersR (TupRsingle scalarTypeInt)
    , DeclareVars lhs''' w''' k''' <- declareVars $ buffersR $ shapeType sh
    = aletUnique lhs (Compute (ShapeSize sh (paramsIn' $ fromGrounds gv))) $
      -- 1) Create a Tensor of flattened shape sh with only ones
      aletUnique lhs' (desugarAlloc (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars w $ fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        (TConstant scalarTypeInt 1)
        (ArgArray Out (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars (w' .> w) gv) (k' weakenId) :>: ArgsNil)
      ) $
      -- 2) Obtain a tensor of indices (tf.where returns list of indices pointing to values > 0)
      aletUnique lhs'' (desugarAlloc (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars (w' .> w) $ fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        TWhere
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeInt)) (TupRpair TupRunit (k (w'' .> w'))) (k' w'') :>:
         ArgArray Out (ArrayR dim1 (TupRsingle scalarTypeInt)) (TupRpair TupRunit (k (w'' .> w'))) (k'' weakenId) :>:
         ArgsNil)
      ) $
      -- 3) Convert 1d indices to multidimensional indices
      aletUnique lhs''' (desugarAlloc (ArrayR dim1 (shapeType sh)) (fromGrounds $ TupRpair TupRunit (k (w'' .> w')))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkMap
        (ArgFun $ Lam (LeftHandSideSingle scalarTypeInt) $ Body (FromIndex sh (paramsIn' $ fromGrounds $ weakenVars (w''' .> w'' .> w' .> w) gv) (Evar (Var scalarTypeInt ZeroIdx))))
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeInt)) (TupRpair TupRunit (k (w''' .> w'' .> w'))) (k'' w'''))
        (ArgArray Out (ArrayR dim1 (shapeType sh)) (TupRpair TupRunit (k (w''' .> w'' .> w'))) (k''' weakenId))
      )
      -- 4) Apply f to the indices
      (mkMap
        (weaken (w''' .> w'' .> w' .> w) f)
        (ArgArray In (ArrayR dim1 (shapeType sh)) (TupRpair TupRunit (k (w''' .> w'' .> w'))) (k''' weakenId))
        (ArgArray Out (ArrayR dim1 t) (TupRpair TupRunit (k (w''' .> w'' .> w'))) (weakenVars (w''' .> w'' .> w' .> w) gvb))
      )

  mkPermute :: Arg env (Fun' (e -> e -> e)) -> Arg env (Mut sh' e) -> Arg env (Fun' (sh -> PrimMaybe sh')) -> Arg env (In sh e) -> OperationAcc TensorOp env ()
  mkPermute
    (ArgFun comb)
    (ArgArray _ (ArrayR sh' _) gv' gvb')
    perm
    (ArgArray _ (ArrayR sh t) gv gvb)
    | maybeSh'                     <- TupRpair (TupRsingle scalarTypeWord8) (TupRpair TupRunit (shapeType sh'))
    , DeclareVars lhs w k          <- declareVars $ buffersR maybeSh'
    , DeclareVars lhs' w' k'       <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
    , DeclareVars lhs'' w'' k''    <- declareVars $ buffersR (shapeType sh')
    , DeclareVars lhs''' w''' k''' <- declareVars $ buffersR t
    = -- 1) Create an array of maybeSh' with perm
      aletUnique lhs (desugarAlloc (ArrayR sh maybeSh') (fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkGenerate
        (weaken w perm)
        (ArgArray Out (ArrayR sh maybeSh') (weakenVars w gv) (k weakenId))
      ) $
      -- 2) To apply boolean mask with 1D arrays, we need to flatten. Calculate the dim1 size.
      aletUnique lhs' (Compute (ShapeSize sh (paramsIn' $ weakenVars w $ fromGrounds gv))) $
      -- 3) Get 1D array of (Just sh'), by applying a boolean mask with predicate isJust.
      aletUnique lhs'' (desugarAlloc (ArrayR sh (shapeType sh')) (fromGrounds (weakenVars (w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask (shapeType sh')
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' w'')) (isJust (TupRpair TupRunit (shapeType sh')) (k (w'' .> w'))))
        (fromJust (shapeType sh') (k (w'' .> w')))
        (k'' weakenId)
      ) $
      -- 3) Get 1D array of source indices (sh) with perm output (Just sh'), by applying a boolean mask with predicate isJust.
      aletUnique lhs''' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w'' .> w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask t
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' (w''' .> w''))) (isJust (TupRpair TupRunit (shapeType sh')) (k (w''' .> w'' .> w'))))
        (weakenVars (w''' .> w'' .> w' .> w) gvb)
        (k''' weakenId)
      ) $
      -- 4) Apply tf.tensor_scatter
      Exec (TTensorScatter scatterFun) (
        ArgArray Mut (ArrayR sh' t) (weakenVars (w''' .> w'' .> w' .> w) gv') (weakenVars (w''' .> w'' .> w' .> w) gvb') :>:
        ArgArray In (ArrayR sh (shapeType sh')) (weakenVars (w''' .> w'' .> w' .> w) gv) (k'' w''') :>:
        ArgArray In (ArrayR sh t) (weakenVars (w''' .> w'' .> w' .> w) gv) (k''' weakenId) :>:
        ArgsNil
      )
        where 
          isJust :: TypeR a -> GroundVars env (Buffers (Word8, a)) -> GroundVars env (Buffers Word8)
          isJust _ (TupRpair word8 _) = word8
          isJust _ _ = error "impossible"

          fromJust :: TypeR a -> GroundVars env (Buffers (Word8, ((), a))) -> GroundVars env (Buffers a)
          fromJust _ (TupRpair _ (TupRpair _ a)) = a
          fromJust _ _ = error "impossible"

          scatterFun = case comb of
            Lam (LeftHandSideSingle _) (Lam (LeftHandSideSingle _) (Body (PrimApp fun (Pair (Evar (Var _ (SuccIdx ZeroIdx))) (Evar (Var _ ZeroIdx)))))) -> case fun of
              PrimAdd _ -> ScatterFunAdd
              PrimSub _ -> ScatterFunMin
              _         -> error "primfun not yet supported"
            _ -> error "complex combination for permute not supported"

mkExp :: forall env env' sh t. BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' t -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkExp _ (Const s e) aOut | OneOfDict <- tfAllDict s = Exec (TConstant s e) $ aOut :>: ArgsNil

mkExp env (PrimApp f exp) aOut = mkPrimFun f env exp aOut

mkExp env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp1
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkExp
     (weakenEnv w env)
     (weakenArrayInstr w exp1)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   mkExp
     (push' (weakenEnv w env) (elhs, distributeBIdx a (k weakenId)))
     (weakenArrayInstr w exp2)
     (weaken w aOut)

mkExp env (Evar (Var st idx)) (ArgArray _ arrayR gv gvb@(TupRsingle (Var groundR _)))
  | Refl <- reprIsSingle @ScalarType @t @Buffer st
  , OneOfDict <- tfAllDict st
  = let (BIdx idx') = prj' idx env
        gvb'        = TupRsingle (Var groundR idx')
    in  Exec
          (TId st)
          (
            ArgArray In arrayR gv gvb' :>:
            ArgArray Out arrayR gv gvb :>:
            ArgsNil
          )

mkExp env (Pair exp1 exp2) (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2))
 = Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkExp env exp1 (ArgArray Out (ArrayR sh t1) gv gvb1))
   (mkExp env exp2 (ArgArray Out (ArrayR sh t2) gv gvb2))

mkExp env (ToIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ shapeType sh
  = mkExp env (Let lhs exp1 (Let lhs' (weakenE w exp2) (toIndex sh (k w') (k' weakenId)))) aOut

mkExp env (ShapeSize sh exp) aOut
  | DeclareVars lhs _ k <- declareVars $ shapeType sh
  = mkExp env (Let lhs exp (shapeSize sh (k weakenId))) aOut

mkExp env (FromIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ TupRsingle scalarTypeInt
  = mkExp env (Let lhs exp1 (Let lhs' (weakenE w exp2) (fromIndex sh (k w') (k' weakenId)))) aOut

mkExp env (IndexSlice slix exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ sliceEltR slix
  , DeclareVars lhs' w' k' <- declareVars $ shapeType (sliceDomainR slix)
  = mkExp env (Let lhs exp1 (Let lhs' (weakenE w exp2) (indexSlice slix (k w') (k' weakenId)))) aOut

mkExp env (IndexFull slix exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ sliceEltR slix
  , DeclareVars lhs' w' k' <- declareVars $ shapeType (sliceShapeR slix)
  = mkExp env (Let lhs exp1 (Let lhs' (weakenE w exp2) (indexFull slix (k w') (k' weakenId)))) aOut

mkExp env (Cond cond exp1 exp2) (ArgArray _ (ArrayR sh t) gv gvb)
  -- -| isNotLoop exp1 -- todo add preprocess check to throw an error if theres a loop, mem access, etc.
  -- , isNotLoop exp2
  | DeclareVars lhs w k <- declareVars $ buffersR $ TupRsingle scalarTypeWord8
  , DeclareVars lhs' w' k' <- declareVars $ buffersR t
  , DeclareVars lhs'' w'' k'' <- declareVars $ buffersR t
  = aletUnique lhs (desugarAlloc (ArrayR sh (TupRsingle scalarTypeWord8)) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv w env) (weakenArrayInstr w cond) (ArgArray Out (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars w gv) (k weakenId))) $
    aletUnique lhs' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars w gv))) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv (w' .> w) env) (weakenArrayInstr (w' .> w) exp1) (ArgArray Out (ArrayR sh t) (weakenVars (w' .> w) gv) (k' weakenId))) $
    aletUnique lhs'' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w' .> w) gv))) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv (w'' .> w' .> w) env) (weakenArrayInstr (w'' .> w' .> w) exp2) (ArgArray Out (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId)))
    (select
      t
      (ArgArray In (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars (w'' .> w' .> w) gv) (k (w'' .> w')))
      (ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k' w''))
      (ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId))
      (ArgArray Out (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (weakenVars (w'' .> w' .> w) gvb))
    )

mkExp _ Nil _ = Return TupRunit

mkExp env (ArrayInstr (Index var) exp) (ArgArray _ (ArrayR sh t@(TupRsingle st)) gv gvb)
  | Refl <- reprIsSingle @ScalarType @t @Buffer st
  , OneOfDict <- tfAllDict st
  , i <- expType exp
  , DeclareVars lhs w k <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
  , DeclareVars lhs' w' k' <- declareVars $ buffersR i
  =  -- To apply gather to 1d array, calculate the dim1 size
    aletUnique lhs (Compute (ShapeSize sh (paramsIn' $ fromGrounds gv))) $
    aletUnique lhs' (desugarAlloc (ArrayR sh i) (weakenVars w $ fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv (w' .> w) env) (weakenArrayInstr (w' .> w) exp) (ArgArray Out (ArrayR sh i) (weakenVars (w' .> w) gv) (k' weakenId)))
    (Exec
      (TGather st)
      (
        ArgArray In (ArrayR dim1 t) (TupRpair TupRunit (k w')) (TupRsingle (weaken (w' .> w) var)) :>:
        ArgArray In (ArrayR sh i) (weakenVars (w' .> w) gv) (k' weakenId) :>:
        ArgArray Out (ArrayR sh t) (weakenVars (w' .> w) gv) (weakenVars (w' .> w) gvb) :>:
        ArgsNil
      )
    )

mkExp _ (ArrayInstr (Parameter var@(Var st _)) Nil) aOut
  | OneOfDict <- tfAllDict st
  = Exec
      (TVar st)
      (
        ArgVar (TupRsingle var) :>:
        aOut :>:
        ArgsNil
      )

mkExp env (Undef st) aOut = mkExp env (Const st (zero st)) aOut
mkExp env (Coerce stIn stOut exp) (ArgArray _ (ArrayR sh t) gv gvb)
  | OneOfDict <- tfAllDict stIn
  , OneOfDict <- tfAllDict stOut
  , a <- expType exp
  , DeclareVars lhs w k <- declareVars $ buffersR a
  = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv w env) (weakenArrayInstr w exp) (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId)))
    (Exec
      (TCast stIn stOut)
      ( ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
        ArgArray Out (ArrayR sh t) (weakenVars w gv) (weakenVars w gvb) :>:
        ArgsNil)
     )

mkExp env (PrimConst (PrimMinBound bt@(IntegralBoundedType it))) aOut = mkExp env (Const (SingleScalarType (NumSingleType (IntegralNumType it))) (evalMinBound bt)) aOut
mkExp env (PrimConst (PrimMaxBound bt@(IntegralBoundedType it))) aOut = mkExp env (Const (SingleScalarType (NumSingleType (IntegralNumType it))) (evalMaxBound bt)) aOut
mkExp env (PrimConst (PrimPi ft)) aOut                                = mkExp env (Const (SingleScalarType (NumSingleType (FloatingNumType ft))) (evalPi ft)) aOut

mkExp env (Foreign _ _ fallback exp) (ArgArray _ (ArrayR sh t) gv gvb)
  | a <- expType exp
  , DeclareVars lhs w k <- declareVars $ buffersR a
  = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv w env) (weakenArrayInstr w exp) (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId)))
    (mkMap 
      (ArgFun (rebuildNoArrayInstr fallback)) 
      (ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId))
      (ArgArray Out (ArrayR sh t) (weakenVars w gv) (weakenVars w gvb)))

-- Not supported
mkExp _ VecPack {} _   = error "VecPack operation not supported by TensorFlow backend."
mkExp _ VecUnpack {} _ = error "VecUnpack operation not supported by TensorFlow backend."
mkExp _ Case {} _      = error "Case operation not supported by TensorFlow backend."
mkExp _ While {} _     = error "While operation not supported by TensorFlow backend."

mkExp _ _ _ = error "impossible"

select :: TypeR t -> Arg env (In sh Word8) -> Arg env (In sh t) -> Arg env (In sh t) -> Arg env (Out sh t) -> PreOpenAcc TensorOp env ()
select (TupRpair t1 t2) (ArgArray _ (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray _ _ gvIn2 (TupRpair gvbIn21 gvbIn22)) (ArgArray _ _ gvIn3 (TupRpair gvbIn31 gvbIn32)) (ArgArray _ _ gvOut (TupRpair gvbOut1 gvbOut2)) =
  Alet (LeftHandSideWildcard TupRunit) TupRunit
  (select t1 (ArgArray In (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray In (ArrayR sh t1) gvIn2 gvbIn21) (ArgArray In (ArrayR sh t1) gvIn3 gvbIn31) (ArgArray Out (ArrayR sh t1) gvOut gvbOut1))
  (select t2 (ArgArray In (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray In (ArrayR sh t2) gvIn2 gvbIn22) (ArgArray In (ArrayR sh t2) gvIn3 gvbIn32) (ArgArray Out (ArrayR sh t2) gvOut gvbOut2))
select t@(TupRsingle s) (ArgArray _ (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray _ _ _ gvbIn2) (ArgArray _ _ _ gvbIn3) (ArgArray _ _ gvOut gvbOut)
  | OneOfDict <- tfAllDict s
  = Exec
    (TSelect s)
    (ArgArray In (ArrayR sh (TupRpair tWord8 (TupRpair (TupRsingle s) (TupRsingle s)))) gvIn1 (TupRpair gvbIn1 (TupRpair gvbIn2 gvbIn3)) :>:
     ArgArray Out (ArrayR sh t) gvOut gvbOut :>:
     ArgsNil)
select TupRunit _ _ _ _ = Return TupRunit
select _ _ _ _ _ = error "impossible"

mkPrimFun :: PrimFun (a -> t) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkPrimFun (PrimAdd nt)  | OneOfDict <- tfNumDict nt = mkPrimFun'  $ TAdd (SingleScalarType (NumSingleType nt))
mkPrimFun (PrimMul nt)  | OneOfDict <- tfNumDict nt = mkPrimFun'  $ TMul (SingleScalarType (NumSingleType nt))
mkPrimFun (PrimSub nt)  | OneOfDict <- tfNumDict nt = mkPrimFun'  $ TSub (SingleScalarType (NumSingleType nt))
mkPrimFun (PrimNeg nt)  | OneOfDict <- tfNum'Dict nt = mkPrimFun' $ TNeg (SingleScalarType (NumSingleType nt))
mkPrimFun (PrimAbs nt)  | OneOfDict <- tfNum'Dict nt = mkPrimFun' $ TAbs (SingleScalarType (NumSingleType nt))
mkPrimFun (PrimSig nt)  | OneOfDict <- tfNum'Dict nt = mkPrimFun' $ TSign (SingleScalarType (NumSingleType nt))

mkPrimFun (PrimQuot it) | OneOfDict <- tfNumDict (IntegralNumType it) = mkPrimFun' $ TTruncateDiv (SingleScalarType (NumSingleType (IntegralNumType it)))
mkPrimFun (PrimRem it)  | OneOfDict <- tfModDict it = mkPrimFun' $ TTruncateMod (SingleScalarType (NumSingleType (IntegralNumType it)))
mkPrimFun (PrimQuotRem it) = mkPrimFun2' (PrimQuot it) (PrimRem it)

mkPrimFun (PrimIDiv it) | OneOfDict <- tfNumDict (IntegralNumType it) = mkPrimFun' $ TRealDiv (SingleScalarType (NumSingleType (IntegralNumType it)))
mkPrimFun (PrimMod it)  | OneOfDict <- tfModDict it = mkPrimFun' $ TTruncateMod (SingleScalarType (NumSingleType (IntegralNumType it)))

mkPrimFun (PrimDivMod it) = mkPrimFun2' (PrimIDiv it) (PrimMod it)

mkPrimFun (PrimBAnd it) | OneOfDict <- tfIntDict it = mkPrimFun' $ TBitwiseAnd (SingleScalarType (NumSingleType (IntegralNumType it)))
mkPrimFun (PrimBOr it)  | OneOfDict <- tfIntDict it = mkPrimFun' $ TBitwiseOr (SingleScalarType (NumSingleType (IntegralNumType it)))
mkPrimFun (PrimBXor it) | OneOfDict <- tfIntDict it = mkPrimFun' $ TBitwiseXor (SingleScalarType (NumSingleType (IntegralNumType it)))
mkPrimFun (PrimBNot it) | OneOfDict <- tfIntDict it = mkPrimFun' $ TInvert (SingleScalarType (NumSingleType (IntegralNumType it)))
mkPrimFun (PrimBShiftL it) | OneOfDict <- tfIntDict it = undefined
mkPrimFun (PrimBShiftR it) | OneOfDict <- tfIntDict it = undefined
mkPrimFun (PrimBRotateL it) | OneOfDict <- tfIntDict it = undefined
mkPrimFun (PrimBRotateR it) | OneOfDict <- tfIntDict it = undefined
mkPrimFun (PrimPopCount it) | OneOfDict <- tfIntDict it = undefined
mkPrimFun (PrimCountLeadingZeros it) | OneOfDict <- tfIntDict it = undefined
mkPrimFun (PrimCountTrailingZeros it) | OneOfDict <- tfIntDict it = undefined

mkPrimFun (PrimFDiv ft) | OneOfDict <- tfNumDict (FloatingNumType ft) = mkPrimFun' $ TRealDiv (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimRecip ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TReciprocal (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimSin ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TSin (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimCos ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TCos (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimTan ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TTan (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimAsin ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TAsin (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimAcos ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TAcos (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimAtan ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TAtan (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimSinh ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TSinh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimCosh ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TCosh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimTanh ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TTanh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimAsinh ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TAsinh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimAcosh ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TAcosh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimAtanh ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TAtanh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimExpFloating ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TExp (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimSqrt ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TSqrt (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimLog ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TLog (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimFPow ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TPow (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimLogBase ft) | OneOfDict <- tfFloatDict ft = undefined -- mkPrimFun' $ TLog1p (SingleScalarType (NumSingleType (FloatingNumType ft)))

mkPrimFun (PrimTruncate ta tb) = undefined
mkPrimFun (PrimRound ta tb) = undefined
mkPrimFun (PrimFloor ta tb) = undefined
mkPrimFun (PrimCeiling ta tb) = undefined

mkPrimFun (PrimAtan2 ft) | OneOfDict <- tfFloatDict ft = mkPrimFun' $ TAtan2 (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkPrimFun (PrimIsNaN _) = undefined
mkPrimFun (PrimIsInfinite _) = undefined

mkPrimFun (PrimLt st) | OneOfDict <- tfOrdDict st = mkPrimFun' $ TLess (SingleScalarType st)
mkPrimFun (PrimGt st) | OneOfDict <- tfOrdDict st = mkPrimFun' $ TGreater (SingleScalarType st)
mkPrimFun (PrimLtEq st) | OneOfDict <- tfOrdDict st = mkPrimFun' $ TLessEqual (SingleScalarType st)
mkPrimFun (PrimGtEq st) | OneOfDict <- tfOrdDict st = mkPrimFun' $ TGreaterEqual (SingleScalarType st)
mkPrimFun (PrimEq st) | OneOfDict <- tfOrdDict st = mkPrimFun' $ TEqual (SingleScalarType st)
mkPrimFun (PrimNEq st) | OneOfDict <- tfOrdDict st = mkPrimFun' $ TNotEqual (SingleScalarType st)
mkPrimFun (PrimMax st) | OneOfDict <- tfOrdDict st = mkPrimFun' $ TMaximum (SingleScalarType st)
mkPrimFun (PrimMin st) | OneOfDict <- tfOrdDict st = mkPrimFun' $ TMinimum (SingleScalarType st)

mkPrimFun PrimLAnd = mkPrimFun' TLogicalAnd
mkPrimFun PrimLOr = mkPrimFun' TLogicalOr
mkPrimFun PrimLNot = mkPrimFun' TLogicalNot

mkPrimFun (PrimFromIntegral it nt)
  | TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType it)))
  , TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType nt))
  = mkPrimFun' $ TCast (SingleScalarType (NumSingleType (IntegralNumType it))) (SingleScalarType (NumSingleType nt))

mkPrimFun (PrimToFloating _ _) = undefined

mkPrimFun' :: TensorOp (In sh a -> Out sh b -> ()) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh b) -> OperationAcc TensorOp env ()
mkPrimFun' op env exp aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkExp -- flatten higher-order expression
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   Exec -- apply method to the result
    op
    (
      ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
      weaken w aOut :>:
      ArgsNil
    )

mkPrimFun2' :: PrimFun ((a, a) -> a) -> PrimFun ((a, a) -> a) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' (a, a) -> Arg env (Out sh (a, a)) -> OperationAcc TensorOp env ()
mkPrimFun2' fun1 fun2 env exp (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2)) =
  Alet (LeftHandSideWildcard TupRunit) TupRunit
  (mkPrimFun
    fun1
    env
    exp
    (ArgArray Out (ArrayR sh t1) gv gvb1)
  )
  (mkPrimFun
    fun2
    env
    exp
    (ArgArray Out (ArrayR sh t2) gv gvb2)
  )
mkPrimFun2' _ _ _ _ _ = error "impossible"

booleanMask :: TypeR a -> Arg env (In DIM1 Word8) -> GroundVars env (Buffers a) -> GroundVars env (Buffers a) -> PreOpenAcc TensorOp env ()
booleanMask TupRunit _ _ gvb = Return gvb
booleanMask t@(TupRsingle s) (ArgArray _ _ gv gvbIn2) gvbIn1 gvbOut
  | OneOfDict <- tfAllDict s
  = Exec (TBooleanMask s) (ArgArray In (ArrayR dim1 (TupRpair t (TupRsingle scalarTypeWord8))) gv (TupRpair gvbIn1 gvbIn2) :>: ArgArray Out (ArrayR dim1 t) gv gvbOut :>: ArgsNil)
booleanMask (TupRpair t1 t2) aOut (TupRpair gvbIn1 gvbIn2) (TupRpair gvbOut1 gvbOut2) = Alet (LeftHandSideWildcard TupRunit) TupRunit
 (booleanMask t1 aOut gvbIn1 gvbOut1)
 (booleanMask t2 aOut gvbIn2 gvbOut2)
booleanMask _ _ _ _ = error "impossible"

shapeSize :: ShapeR sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
shapeSize ShapeRz TupRunit                                = Const scalarTypeInt 1
shapeSize (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz))
  = PrimApp (PrimMul (IntegralNumType TypeInt))
      (Pair
        (shapeSize shr sh)
        (Evar sz))
shapeSize _ _                                             = error "impossible"

toIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
toIndex ShapeRz TupRunit TupRunit = Const scalarTypeInt 0
toIndex (ShapeRsnoc shr) (TupRpair sh _) (TupRpair ix (TupRsingle i))
  = PrimApp (PrimAdd (IntegralNumType TypeInt))
      (Pair
        (toIndex shr sh ix)
        (PrimApp (PrimMul (IntegralNumType TypeInt))
          (Pair
            (shapeSize shr sh)
            (Evar i))))
toIndex _ _ _                     = error "impossible"

fromIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' Int -> PreOpenExp (ArrayInstr env) env' sh
fromIndex ShapeRz TupRunit _                                = Nil
fromIndex (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz)) (TupRsingle i)
  | DeclareVars lhs w k <- declareVars $ TupRsingle scalarTypeInt
  = Pair
    (Let
      lhs
      (PrimApp (PrimQuot TypeInt) (Pair (Evar i) (Evar sz)))
      (fromIndex shr (weakenVars w sh) (k weakenId)))
    (case shr of
      ShapeRz -> Evar i
      _       -> PrimApp (PrimRem TypeInt) (Pair (Evar i) (Evar sz))
    )
fromIndex _ _ _              = error "impossible"

indexSlice :: SliceIndex slix t co sh1 -> ExpVars env' slix -> ExpVars env' sh1 -> PreOpenExp (ArrayInstr env) env' t
indexSlice SliceNil _ _ = Nil
indexSlice (SliceAll slix) (TupRpair slx TupRunit) (TupRpair sl (TupRsingle sz)) =
  Pair (indexSlice slix slx sl) (Evar sz)
indexSlice (SliceFixed slix) (TupRpair slx _) (TupRpair sl _) =
  indexSlice slix slx sl
indexSlice _ _ _ = error "impossible"

indexFull :: SliceIndex slix sl co t -> ExpVars env' slix -> ExpVars env' sl -> PreOpenExp (ArrayInstr env) env' t
indexFull SliceNil _ _ = Nil
indexFull (SliceAll slix) (TupRpair slx TupRunit) (TupRpair sl (TupRsingle sz)) =
  Pair (indexFull slix slx sl) (Evar sz)
indexFull (SliceFixed slix) (TupRpair slz (TupRsingle sz)) sl =
  Pair (indexFull slix slz sl) (Evar sz)
indexFull _ _ _ = error "impossible"

weakenEnv :: Sink f => (env :> env') -> Env (f env) a -> Env (f env') a
weakenEnv w = mapEnv (weaken w)

distributeBIdx :: forall env s. TypeR s -> GroundVars env (Buffers s) -> Distribute (BufferIdx env) s
distributeBIdx TupRunit _ = ()
distributeBIdx (TupRsingle s) (TupRsingle (Var _ idx))
  | Refl <- reprIsSingle @ScalarType @s @(BufferIdx env) s
  , Refl <- reprIsSingle @ScalarType @s @Buffer s
  = BIdx idx
distributeBIdx (TupRpair l1 l2) (TupRpair r1 r2) = (distributeBIdx l1 r1, distributeBIdx l2 r2)
distributeBIdx _ _ = error "impossible"
